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
def rename_key_and_reshape_tensor(pt_tuple_key: Tuple[str], pt_tensor: np.ndarray, random_flax_state_dict: Dict[str, jnp.ndarray], model_prefix: str) -> (Tuple[str], np.ndarray):
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""

    def is_key_or_prefix_key_in_dict(key: Tuple[str]) -> bool:
        """Checks if `key` of `(prefix,) + key` is in random_flax_state_dict"""
        return len(set(random_flax_state_dict) & {key, (model_prefix,) + key}) > 0
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('scale',)
    if pt_tuple_key[-1] in ['weight', 'gamma'] and is_key_or_prefix_key_in_dict(renamed_pt_tuple_key):
        return (renamed_pt_tuple_key, pt_tensor)
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('mean',)
    if pt_tuple_key[-1] == 'running_mean' and (not is_key_or_prefix_key_in_dict(pt_tuple_key)):
        return (renamed_pt_tuple_key, pt_tensor)
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('var',)
    if pt_tuple_key[-1] == 'running_var' and (not is_key_or_prefix_key_in_dict(pt_tuple_key)):
        return (renamed_pt_tuple_key, pt_tensor)
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('embedding',)
    if pt_tuple_key[-1] == 'weight' and is_key_or_prefix_key_in_dict(renamed_pt_tuple_key):
        return (renamed_pt_tuple_key, pt_tensor)
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('kernel',)
    if pt_tuple_key[-1] == 'weight' and pt_tensor.ndim == 4 and (not is_key_or_prefix_key_in_dict(pt_tuple_key)):
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return (renamed_pt_tuple_key, pt_tensor)
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('kernel',)
    if pt_tuple_key[-1] == 'weight' and (not is_key_or_prefix_key_in_dict(pt_tuple_key)):
        pt_tensor = pt_tensor.T
        return (renamed_pt_tuple_key, pt_tensor)
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('weight',)
    if pt_tuple_key[-1] == 'gamma':
        return (renamed_pt_tuple_key, pt_tensor)
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ('bias',)
    if pt_tuple_key[-1] == 'beta':
        return (renamed_pt_tuple_key, pt_tensor)
    name = None
    if pt_tuple_key[-3::2] == ('parametrizations', 'original0'):
        name = pt_tuple_key[-2] + '_g'
    elif pt_tuple_key[-3::2] == ('parametrizations', 'original1'):
        name = pt_tuple_key[-2] + '_v'
    if name is not None:
        renamed_pt_tuple_key = pt_tuple_key[:-3] + (name,)
        return (renamed_pt_tuple_key, pt_tensor)
    return (pt_tuple_key, pt_tensor)