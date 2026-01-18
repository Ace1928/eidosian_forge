import json
import logging
import os
import platform
from abc import ABCMeta, abstractmethod
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from packaging import version
import ray
import ray.cloudpickle as pickle
from ray.actor import ActorHandle
from ray.train import Checkpoint
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import (
from ray.rllib.utils.checkpoints import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.serialization import (
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.tensor_dtype import get_np_dtype
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@ExperimentalAPI
def learn_on_batch_from_replay_buffer(self, replay_actor: ActorHandle, policy_id: PolicyID) -> Dict[str, TensorType]:
    """Samples a batch from given replay actor and performs an update.

        Args:
            replay_actor: The replay buffer actor to sample from.
            policy_id: The ID of this policy.

        Returns:
            Dictionary of extra metadata from `compute_gradients()`.
        """
    batch = ray.get(replay_actor.replay.remote(policy_id=policy_id))
    if batch is None:
        return {}
    if hasattr(self, 'devices') and len(self.devices) > 1:
        self.load_batch_into_buffer(batch, buffer_index=0)
        return self.learn_on_loaded_batch(offset=0, buffer_index=0)
    else:
        return self.learn_on_batch(batch)