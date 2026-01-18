import argparse
import json
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.models import create_model
from transformers import (
def load_beit_model(args, is_finetuned, is_large):

    def load_state_dict(model, state_dict, prefix='', ignore_missing='relative_position_index'):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix=prefix)
        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split('|'):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)
        missing_keys = warn_missing_keys
        if len(missing_keys) > 0:
            print('Weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print('Weights from pretrained model not used in {}: {}'.format(model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print('Ignored weights of {} not initialized from pretrained model: {}'.format(model.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs))
    model_kwargs = {'pretrained': False, 'use_shared_rel_pos_bias': True, 'use_abs_pos_emb': False, 'init_values': 0.1}
    if is_finetuned:
        model_kwargs.update({'num_classes': 1000, 'use_mean_pooling': True, 'init_scale': 0.001, 'use_rel_pos_bias': True})
    model = create_model('beit_large_patch16_224' if is_large else 'beit_base_patch16_224', **model_kwargs)
    patch_size = model.patch_embed.patch_size
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    checkpoint = torch.load(args.beit_checkpoint, map_location='cpu')
    print(f'Load ckpt from {args.beit_checkpoint}')
    checkpoint_model = None
    for model_key in ('model', 'module'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print(f'Load state_dict by model_key = {model_key}')
            break
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if 'relative_position_index' in key:
            checkpoint_model.pop(key)
        if 'relative_position_bias_table' in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
    load_state_dict(model, checkpoint_model, prefix='')
    return model