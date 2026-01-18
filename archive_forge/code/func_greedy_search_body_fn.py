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
def greedy_search_body_fn(state):
    """state update fn."""
    model_outputs = model(state.running_token, params=params, **state.model_kwargs)
    logits = model_outputs.logits[:, -1]
    logits = logits_processor(state.sequences, logits, state.cur_len)
    next_token = jnp.argmax(logits, axis=-1)
    next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
    next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
    next_token = next_token[:, None]
    next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
    next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)
    return GreedyState(cur_len=state.cur_len + 1, sequences=next_sequences, running_token=next_token, is_sent_finished=next_is_sent_finished, model_kwargs=next_model_kwargs)