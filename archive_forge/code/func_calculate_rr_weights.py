import logging
from typing import List, Optional, Type, Callable
import numpy as np
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.simple_q.simple_q import (
from ray.rllib.execution.rollout_ops import (
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.execution.train_ops import (
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.metrics import (
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.replay_buffers.utils import sample_min_n_steps_from_buffer
def calculate_rr_weights(config: AlgorithmConfig) -> List[float]:
    """Calculate the round robin weights for the rollout and train steps"""
    if not config['training_intensity']:
        return [1, 1]
    native_ratio = config['train_batch_size'] / (config.get_rollout_fragment_length() * config['num_envs_per_worker'] * max(config['num_workers'] + 1, 1))
    sample_and_train_weight = config['training_intensity'] / native_ratio
    if sample_and_train_weight < 1:
        return [int(np.round(1 / sample_and_train_weight)), 1]
    else:
        return [1, int(np.round(sample_and_train_weight))]