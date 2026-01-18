import gymnasium as gym
import numpy as np
import tree
from typing import Dict, Any, List
import logging
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.utils.policy import compute_log_likelihoods_from_input_dict
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TensorType, SampleBatchType
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
@DeveloperAPI
def check_action_prob_in_batch(self, batch: SampleBatchType) -> None:
    """Checks if we support off policy estimation (OPE) on given batch.

        Args:
            batch: The batch to check.

        Raises:
            ValueError: In case `action_prob` key is not in batch
        """
    if 'action_prob' not in batch:
        raise ValueError("Off-policy estimation is not possible unless the inputs include action probabilities (i.e., the policy is stochastic and emits the 'action_prob' key). For DQN this means using `exploration_config: {type: 'SoftQ'}`. You can also set `off_policy_estimation_methods: {}` to disable estimation.")