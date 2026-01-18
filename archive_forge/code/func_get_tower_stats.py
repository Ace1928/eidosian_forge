import copy
import functools
import logging
import math
import os
import threading
import time
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import ray
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import NullContextManager, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.error import ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
@DeveloperAPI
def get_tower_stats(self, stats_name: str) -> List[TensorStructType]:
    """Returns list of per-tower stats, copied to this Policy's device.

        Args:
            stats_name: The name of the stats to average over (this str
                must exist as a key inside each tower's `tower_stats` dict).

        Returns:
            The list of stats tensor (structs) of all towers, copied to this
            Policy's device.

        Raises:
            AssertionError: If the `stats_name` cannot be found in any one
            of the tower's `tower_stats` dicts.
        """
    data = []
    for tower in self.model_gpu_towers:
        if stats_name in tower.tower_stats:
            data.append(tree.map_structure(lambda s: s.to(self.device), tower.tower_stats[stats_name]))
    assert len(data) > 0, f'Stats `{stats_name}` not found in any of the towers (you have {len(self.model_gpu_towers)} towers in total)! Make sure you call the loss function on at least one of the towers.'
    return data