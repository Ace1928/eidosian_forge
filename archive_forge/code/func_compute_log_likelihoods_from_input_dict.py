import gymnasium as gym
import logging
import numpy as np
import re
from typing import (
import tree  # pip install dm_tree
import ray.cloudpickle as pickle
from ray.rllib.models.preprocessors import ATARI_OBS_SHAPE
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import (
from ray.util import log_once
from ray.util.annotations import PublicAPI
@PublicAPI
def compute_log_likelihoods_from_input_dict(policy: 'Policy', batch: Union[SampleBatch, Dict[str, TensorStructType]]):
    """Returns log likelihood for actions in given batch for policy.

    Computes likelihoods by passing the observations through the current
    policy's `compute_log_likelihoods()` method

    Args:
        batch: The SampleBatch or MultiAgentBatch to calculate action
            log likelihoods from. This batch/batches must contain OBS
            and ACTIONS keys.

    Returns:
        The probabilities of the actions in the batch, given the
        observations and the policy.
    """
    num_state_inputs = 0
    for k in batch.keys():
        if k.startswith('state_in_'):
            num_state_inputs += 1
    state_keys = ['state_in_{}'.format(i) for i in range(num_state_inputs)]
    log_likelihoods: TensorType = policy.compute_log_likelihoods(actions=batch[SampleBatch.ACTIONS], obs_batch=batch[SampleBatch.OBS], state_batches=[batch[k] for k in state_keys], prev_action_batch=batch.get(SampleBatch.PREV_ACTIONS), prev_reward_batch=batch.get(SampleBatch.PREV_REWARDS), actions_normalized=policy.config.get('actions_in_input_normalized', False))
    return log_likelihoods