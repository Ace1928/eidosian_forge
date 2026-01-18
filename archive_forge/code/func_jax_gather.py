from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_big_bird import BigBirdConfig
@staticmethod
def jax_gather(params, indices, batch_dims=2):
    """
        Gather the indices from params correctly (equivalent to tf.gather but with modifications)

        Args:
            params: (bsz, n_heads, num_blocks, block_size, head_dim)
            indices: (<num_blocks, 1)
        """

    def _jax_gather(params, indices):
        return params[indices]
    for _ in range(batch_dims):
        _jax_gather = jax.vmap(_jax_gather, in_axes=(0, 0))
    return _jax_gather(params, indices)