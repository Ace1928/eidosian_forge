import argparse
import os
import re
import torch
from flax.traverse_util import flatten_dict
from t5x import checkpoints
from transformers import (
def rename_and_convert_flax_params(flax_dict):
    converted_dict = {}
    CONVERSION_MAPPING = {'token_embedder': 'embeddings', 'encoder_norm': 'layernorm', 'kernel': 'weight', '.out': '.output', 'scale': 'weight', 'embedders_0.pos_embedding': 'row_embedder.weight', 'embedders_1.pos_embedding': 'column_embedder.weight'}
    DECODER_CONVERSION_MAPPING = {'query': 'attention.query', 'key': 'attention.key', 'value': 'attention.value', 'output.dense': 'output', 'encoder_decoder_attention.o': 'encoder_decoder_attention.attention.o', 'pre_self_attention_layer_norm': 'self_attention.layer_norm', 'pre_cross_attention_layer_norm': 'encoder_decoder_attention.layer_norm', 'mlp.': 'mlp.DenseReluDense.', 'pre_mlp_layer_norm': 'mlp.layer_norm', 'self_attention.o': 'self_attention.attention.o', 'decoder.embeddings.embedding': 'decoder.embed_tokens.weight', 'decoder.relpos_bias.rel_embedding': 'decoder.layer.0.self_attention.attention.relative_attention_bias.weight', 'decoder.decoder_norm.weight': 'decoder.final_layer_norm.weight', 'decoder.logits_dense.weight': 'decoder.lm_head.weight'}
    for key in flax_dict.keys():
        if 'target' in key:
            new_key = '.'.join(key[1:])
            for old, new in CONVERSION_MAPPING.items():
                new_key = new_key.replace(old, new)
            if 'decoder' in new_key:
                for old, new in DECODER_CONVERSION_MAPPING.items():
                    new_key = new_key.replace(old, new)
            if 'layers' in new_key and 'decoder' not in new_key:
                new_key = re.sub('layers_(\\d+)', 'layer.\\1', new_key)
                new_key = new_key.replace('encoder', 'encoder.encoder')
            elif 'layers' in new_key and 'decoder' in new_key:
                new_key = re.sub('layers_(\\d+)', 'layer.\\1', new_key)
            converted_dict[new_key] = flax_dict[key]
    converted_torch_dict = {}
    for key in converted_dict.keys():
        if 'embed_tokens' not in key and 'embedder' not in key:
            converted_torch_dict[key] = torch.from_numpy(converted_dict[key].T)
        else:
            converted_torch_dict[key] = torch.from_numpy(converted_dict[key])
    return converted_torch_dict