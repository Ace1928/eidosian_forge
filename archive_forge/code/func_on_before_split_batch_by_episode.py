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
def on_before_split_batch_by_episode(self, sample_batch: SampleBatch) -> SampleBatch:
    """Called before the batch is split by episode. You can perform any
        preprocessing on the batch that you want here.
        e.g. adding done flags to the batch, or reseting some stats that you want to
        track per episode later during estimation, .etc.

        Args:
            sample_batch: The batch to split by episode. This contains multiple
            episodes.

        Returns:
            The modified batch before calling split_by_episode().
        """
    return sample_batch