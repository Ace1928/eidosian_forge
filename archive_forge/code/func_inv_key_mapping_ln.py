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
def inv_key_mapping_ln(key):
    key = re.sub('bert.emb_ln.', 'bert.embeddings.LayerNorm.', key)
    key = re.sub('bert.encoder.layers.(\\d+).norm1.(weight|bias)', 'bert.encoder.layers.\\1.attention.output.LayerNorm.\\2', key)
    key = re.sub('bert.encoder.layers.(\\d+).norm2.(weight|bias)', 'bert.encoder.layers.\\1.output.LayerNorm.\\2', key)
    key = re.sub('cls.predictions.transform.layer_norm.(weight|bias)', 'cls.predictions.transform.LayerNorm.\\1', key)
    return key