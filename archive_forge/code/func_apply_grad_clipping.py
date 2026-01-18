import logging
import os
import warnings
from typing import Dict, List, Optional, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from packaging import version
import ray
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.utils.annotations import Deprecated, PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER
from ray.rllib.utils.typing import (
@PublicAPI
def apply_grad_clipping(policy: 'TorchPolicy', optimizer: LocalOptimizer, loss: TensorType) -> Dict[str, TensorType]:
    """Applies gradient clipping to already computed grads inside `optimizer`.

    Note: This function does NOT perform an analogous operation as
    tf.clip_by_global_norm. It merely clips by norm (per gradient tensor) and
    then computes the global norm across all given tensors (but without clipping
    by that global norm).

    Args:
        policy: The TorchPolicy, which calculated `loss`.
        optimizer: A local torch optimizer object.
        loss: The torch loss tensor.

    Returns:
        An info dict containing the "grad_norm" key and the resulting clipped
        gradients.
    """
    grad_gnorm = 0
    if policy.config['grad_clip'] is not None:
        clip_value = policy.config['grad_clip']
    else:
        clip_value = np.inf
    num_none_grads = 0
    for param_group in optimizer.param_groups:
        params = list(filter(lambda p: p.grad is not None, param_group['params']))
        if params:
            global_norm = nn.utils.clip_grad_norm_(params, clip_value)
            if isinstance(global_norm, torch.Tensor):
                global_norm = global_norm.cpu().numpy()
            grad_gnorm += min(global_norm, clip_value)
        else:
            num_none_grads += 1
    if num_none_grads == len(optimizer.param_groups):
        return {}
    return {'grad_gnorm': grad_gnorm}