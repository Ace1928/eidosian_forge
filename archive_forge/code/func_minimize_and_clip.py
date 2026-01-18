import logging
from typing import Any, Callable, List, Optional, Type, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from ray.rllib.utils.annotations import PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import (
@PublicAPI
def minimize_and_clip(optimizer: LocalOptimizer, objective: TensorType, var_list: List['tf.Variable'], clip_val: float=10.0) -> ModelGradients:
    """Computes, then clips gradients using objective, optimizer and var list.

    Ensures the norm of the gradients for each variable is clipped to
    `clip_val`.

    Args:
        optimizer: Either a shim optimizer (tf eager) containing a
            tf.GradientTape under `self.tape` or a tf1 local optimizer
            object.
        objective: The loss tensor to calculate gradients on.
        var_list: The list of tf.Variables to compute gradients over.
        clip_val: The global norm clip value. Will clip around -clip_val and
            +clip_val.

    Returns:
        The resulting model gradients (list or tuples of grads + vars)
        corresponding to the input `var_list`.
    """
    assert clip_val is None or clip_val > 0.0, clip_val
    if tf.executing_eagerly():
        tape = optimizer.tape
        grads_and_vars = list(zip(list(tape.gradient(objective, var_list)), var_list))
    else:
        grads_and_vars = optimizer.compute_gradients(objective, var_list=var_list)
    return [(tf.clip_by_norm(g, clip_val) if clip_val is not None else g, v) for g, v in grads_and_vars if g is not None]