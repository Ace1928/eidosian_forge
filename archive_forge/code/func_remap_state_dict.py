import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BertConfig, PretrainedConfig
from transformers.models.bert.modeling_bert import (
from flash_attn.bert_padding import (
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import BertEmbeddings
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
from flash_attn.utils.pretrained import state_dict_from_pretrained
def remap_state_dict(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a Huggingface BERT model to be flash_attn compatible.
    """

    def key_mapping_ln_gamma_beta(key):
        key = re.sub('LayerNorm.gamma$', 'LayerNorm.weight', key)
        key = re.sub('LayerNorm.beta$', 'LayerNorm.bias', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln_gamma_beta(k), v) for k, v in state_dict.items()))

    def key_mapping_layers(key):
        return re.sub('^bert.encoder.layer.', 'bert.encoder.layers.', key)
    state_dict = OrderedDict(((key_mapping_layers(k), v) for k, v in state_dict.items()))

    def key_mapping_ln(key):
        key = re.sub('^bert.embeddings.LayerNorm.', 'bert.emb_ln.', key)
        key = re.sub('^bert.encoder.layers.(\\d+).attention.output.LayerNorm.(weight|bias)', 'bert.encoder.layers.\\1.norm1.\\2', key)
        key = re.sub('^bert.encoder.layers.(\\d+).output.LayerNorm.(weight|bias)', 'bert.encoder.layers.\\1.norm2.\\2', key)
        key = re.sub('^cls.predictions.transform.LayerNorm.(weight|bias)', 'cls.predictions.transform.layer_norm.\\1', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))

    def key_mapping_mlp(key):
        key = re.sub('^bert.encoder.layers.(\\d+).intermediate.dense.(weight|bias)', 'bert.encoder.layers.\\1.mlp.fc1.\\2', key)
        key = re.sub('^bert.encoder.layers.(\\d+).output.dense.(weight|bias)', 'bert.encoder.layers.\\1.mlp.fc2.\\2', key)
        return key
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))
    last_layer_subset = getattr(config, 'last_layer_subset', False)
    for d in range(config.num_hidden_layers):
        Wq = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.query.weight')
        Wk = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.key.weight')
        Wv = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.value.weight')
        bq = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.query.bias')
        bk = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.key.bias')
        bv = state_dict.pop(f'bert.encoder.layers.{d}.attention.self.value.bias')
        if not (last_layer_subset and d == config.num_hidden_layers - 1):
            state_dict[f'bert.encoder.layers.{d}.mixer.Wqkv.weight'] = torch.cat([Wq, Wk, Wv], dim=0)
            state_dict[f'bert.encoder.layers.{d}.mixer.Wqkv.bias'] = torch.cat([bq, bk, bv], dim=0)
        else:
            state_dict[f'bert.encoder.layers.{d}.mixer.Wq.weight'] = Wq
            state_dict[f'bert.encoder.layers.{d}.mixer.Wkv.weight'] = torch.cat([Wk, Wv], dim=0)
            state_dict[f'bert.encoder.layers.{d}.mixer.Wq.bias'] = bq
            state_dict[f'bert.encoder.layers.{d}.mixer.Wkv.bias'] = torch.cat([bk, bv], dim=0)

    def key_mapping_attn(key):
        return re.sub('^bert.encoder.layers.(\\d+).attention.output.dense.(weight|bias)', 'bert.encoder.layers.\\1.mixer.out_proj.\\2', key)
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))

    def key_mapping_decoder_bias(key):
        return re.sub('^cls.predictions.bias', 'cls.predictions.decoder.bias', key)
    state_dict = OrderedDict(((key_mapping_decoder_bias(k), v) for k, v in state_dict.items()))
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict['bert.embeddings.word_embeddings.weight']
        state_dict['bert.embeddings.word_embeddings.weight'] = F.pad(word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0]))
        decoder_weight = state_dict['cls.predictions.decoder.weight']
        state_dict['cls.predictions.decoder.weight'] = F.pad(decoder_weight, (0, 0, 0, config.vocab_size - decoder_weight.shape[0]))
        decoder_bias = state_dict['cls.predictions.decoder.bias']
        state_dict['cls.predictions.decoder.bias'] = F.pad(decoder_bias, (0, config.vocab_size - decoder_bias.shape[0]), value=-100.0)
    return state_dict