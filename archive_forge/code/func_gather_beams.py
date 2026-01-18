import copy
import inspect
import warnings
from functools import partial
from typing import Any, Dict, Optional, Union
import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from ..models.auto import (
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .flax_logits_process import (
def gather_beams(nested, beam_indices, batch_size, new_num_beams):
    """
            Gathers the beam slices indexed by beam_indices into new beam array.
            """
    batch_indices = jnp.reshape(jnp.arange(batch_size * new_num_beams) // new_num_beams, (batch_size, new_num_beams))

    def gather_fn(tensor):
        if tensor.ndim == 0:
            return tensor
        else:
            return tensor[batch_indices, beam_indices]
    return jax.tree_util.tree_map(gather_fn, nested)