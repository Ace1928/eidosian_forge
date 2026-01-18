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
@OverrideToImplementCustomLogic
def on_after_split_batch_by_episode(self, all_episodes: List[SampleBatch]) -> List[SampleBatch]:
    """Called after the batch is split by episode. You can perform any
        postprocessing on each episode that you want here.
        e.g. computing advantage per episode, .etc.

        Args:
            all_episodes: The list of episodes in the original batch. Each element is a
            sample batch type that is a single episode.
        """
    return all_episodes