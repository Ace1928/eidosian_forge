import argparse
import torch
from transformers import NystromformerConfig, NystromformerForMaskedLM
def convert_checkpoint_helper(config, orig_state_dict):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)
        if 'pooler' in key or 'sen_class' in key or 'conv.bias' in key:
            continue
        else:
            orig_state_dict[rename_key(key)] = val
    orig_state_dict['cls.predictions.bias'] = orig_state_dict['cls.predictions.decoder.bias']
    orig_state_dict['nystromformer.embeddings.position_ids'] = torch.arange(config.max_position_embeddings).expand((1, -1)) + 2
    return orig_state_dict