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
def greedy_search_cond_fn(state):
    """state termination condition fn."""
    has_reached_max_length = state.cur_len == max_length
    all_sequence_finished = jnp.all(state.is_sent_finished)
    finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
    return ~finish_generation