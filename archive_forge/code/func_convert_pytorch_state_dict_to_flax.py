import os
from pickle import UnpicklingError
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
import transformers
from . import is_safetensors_available, is_torch_available
from .utils import logging
def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model):
    from_bin = is_torch_available() and isinstance(next(iter(pt_state_dict.values())), torch.Tensor)
    bfloat16 = torch.bfloat16 if from_bin else 'bfloat16'
    weight_dtypes = {k: v.dtype for k, v in pt_state_dict.items()}
    if from_bin:
        for k, v in pt_state_dict.items():
            if v.dtype == bfloat16:
                v = v.float()
            pt_state_dict[k] = v.numpy()
    model_prefix = flax_model.base_model_prefix
    if 'params' in flax_model.params:
        flax_model_params = flax_model.params['params']
    else:
        flax_model_params = flax_model.params
    random_flax_state_dict = flatten_dict(flax_model_params)
    if 'batch_stats' in flax_model.params:
        flax_batch_stats = flatten_dict(flax_model.params['batch_stats'])
        random_flax_state_dict.update(flax_batch_stats)
    flax_state_dict = {}
    load_model_with_head_into_base_model = model_prefix not in flax_model_params and model_prefix in {k.split('.')[0] for k in pt_state_dict.keys()}
    load_base_model_into_model_with_head = model_prefix in flax_model_params and model_prefix not in {k.split('.')[0] for k in pt_state_dict.keys()}
    for pt_key, pt_tensor in pt_state_dict.items():
        pt_tuple_key = tuple(pt_key.split('.'))
        is_bfloat_16 = weight_dtypes[pt_key] == bfloat16
        has_base_model_prefix = pt_tuple_key[0] == model_prefix
        if load_model_with_head_into_base_model and has_base_model_prefix:
            pt_tuple_key = pt_tuple_key[1:]
        flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict, model_prefix)
        require_base_model_prefix = (model_prefix,) + flax_key in random_flax_state_dict
        if load_base_model_into_model_with_head and require_base_model_prefix:
            flax_key = (model_prefix,) + flax_key
        if flax_key in random_flax_state_dict:
            if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                raise ValueError(f'PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape {random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}.')
        if 'batch_stats' in flax_model.params:
            if 'mean' in flax_key[-1] or 'var' in flax_key[-1]:
                flax_state_dict[('batch_stats',) + flax_key] = jnp.asarray(flax_tensor)
                continue
            if 'num_batches_tracked' in flax_key[-1]:
                flax_state_dict.pop(flax_key, None)
                continue
            flax_state_dict[('params',) + flax_key] = jnp.asarray(flax_tensor) if not is_bfloat_16 else jnp.asarray(flax_tensor, dtype=jnp.bfloat16)
        else:
            flax_state_dict[flax_key] = jnp.asarray(flax_tensor) if not is_bfloat_16 else jnp.asarray(flax_tensor, dtype=jnp.bfloat16)
    return unflatten_dict(flax_state_dict)