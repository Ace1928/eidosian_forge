from typing import Callable, List, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_beit import BeitConfig
def relative_position_index_init(window_size: Tuple[int, int]) -> jnp.ndarray:
    """
    get pair-wise relative position index for each token inside the window
    """
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    coords_h = np.arange(window_size[0])
    coords_w = np.arange(window_size[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
    coords_flatten = np.reshape(coords, (2, -1))
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = np.transpose(relative_coords, (1, 2, 0))
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = np.zeros(shape=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return jnp.array(relative_position_index)