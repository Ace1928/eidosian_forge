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
def load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path):
    """Load flax checkpoints in a PyTorch model"""
    flax_checkpoint_path = os.path.abspath(flax_checkpoint_path)
    logger.info(f'Loading Flax weights from {flax_checkpoint_path}')
    flax_cls = getattr(transformers, 'Flax' + model.__class__.__name__)
    if flax_checkpoint_path.endswith('.safetensors'):
        flax_state_dict = safe_load_file(flax_checkpoint_path)
        flax_state_dict = unflatten_dict(flax_state_dict, sep='.')
    else:
        with open(flax_checkpoint_path, 'rb') as state_f:
            try:
                flax_state_dict = from_bytes(flax_cls, state_f.read())
            except UnpicklingError:
                raise EnvironmentError(f'Unable to convert {flax_checkpoint_path} to Flax deserializable object. ')
    return load_flax_weights_in_pytorch_model(model, flax_state_dict)