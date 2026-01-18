from collections import Counter
import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Tuple as GymTuple
import inspect
import logging
import numpy as np
import os
import pprint
import random
import re
import time
import tree  # pip install dm_tree
from typing import (
import yaml
import ray
from ray import air, tune
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune import CLIReporter, run_experiments
def check_off_policyness(results: ResultDict, upper_limit: float, lower_limit: float=0.0) -> Optional[float]:
    """Verifies that the off-policy'ness of some update is within some range.

    Off-policy'ness is defined as the average (across n workers) diff
    between the number of gradient updates performed on the policy used
    for sampling vs the number of gradient updates that have been performed
    on the trained policy (usually the one on the local worker).

    Uses the published DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY metric inside
    a training results dict and compares to the given bounds.

    Note: Only works with single-agent results thus far.

    Args:
        results: The training results dict.
        upper_limit: The upper limit to for the off_policy_ness value.
        lower_limit: The lower limit to for the off_policy_ness value.

    Returns:
        The off-policy'ness value (described above).

    Raises:
        AssertionError: If the value is out of bounds.
    """
    from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
    from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
    learner_info = results['info'][LEARNER_INFO]
    if DEFAULT_POLICY_ID not in learner_info:
        return None
    off_policy_ness = learner_info[DEFAULT_POLICY_ID][DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY]
    if not lower_limit <= off_policy_ness <= upper_limit:
        raise AssertionError(f'`off_policy_ness` ({off_policy_ness}) is outside the given bounds ({lower_limit} - {upper_limit})!')
    return off_policy_ness