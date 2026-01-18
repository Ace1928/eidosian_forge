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
def flatten_beam_dim(tensor):
    """Flattens the first two dimensions of a non-scalar array."""
    if tensor.ndim == 0:
        return tensor
    return tensor.reshape((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])