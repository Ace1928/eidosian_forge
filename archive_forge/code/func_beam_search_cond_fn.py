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
def beam_search_cond_fn(state):
    """beam search state termination condition fn."""
    not_max_length_yet = state.cur_len < max_length
    if early_stopping == 'never' and length_penalty > 0.0:
        best_running_score = state.running_scores[:, :1] / (max_length - decoder_prompt_len) ** length_penalty
    else:
        best_running_score = state.running_scores[:, :1] / (state.cur_len - decoder_prompt_len) ** length_penalty
    worst_finished_score = jnp.where(state.is_sent_finished, jnp.min(state.scores, axis=1, keepdims=True), np.array(-10000000.0))
    improvement_still_possible = jnp.any(best_running_score > worst_finished_score)
    still_open_beam = ~(jnp.all(state.is_sent_finished) & (early_stopping is True))
    return not_max_length_yet & still_open_beam & improvement_still_possible