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
@DeveloperAPI
def export_checkpoint(self, export_dir: str, filename_prefix=DEPRECATED_VALUE, *, policy_state: Optional[PolicyState]=None, checkpoint_format: str='cloudpickle') -> None:
    """Exports Policy checkpoint to a local directory and returns an AIR Checkpoint.

        Args:
            export_dir: Local writable directory to store the AIR Checkpoint
                information into.
            policy_state: An optional PolicyState to write to disk. Used by
                `Algorithm.save_checkpoint()` to save on the additional
                `self.get_state()` calls of its different Policies.
            checkpoint_format: Either one of 'cloudpickle' or 'msgpack'.

        .. testcode::
            :skipif: True

            from ray.rllib.algorithms.ppo import PPOTorchPolicy
            policy = PPOTorchPolicy(...)
            policy.export_checkpoint("/tmp/export_dir")
        """
    if filename_prefix != DEPRECATED_VALUE:
        deprecation_warning(old='Policy.export_checkpoint(filename_prefix=...)', error=True)
    if checkpoint_format not in ['cloudpickle', 'msgpack']:
        raise ValueError(f"Value of `checkpoint_format` ({checkpoint_format}) must either be 'cloudpickle' or 'msgpack'!")
    if policy_state is None:
        policy_state = self.get_state()
    os.makedirs(export_dir, exist_ok=True)
    if checkpoint_format == 'cloudpickle':
        policy_state['checkpoint_version'] = CHECKPOINT_VERSION
        state_file = 'policy_state.pkl'
        with open(os.path.join(export_dir, state_file), 'w+b') as f:
            pickle.dump(policy_state, f)
    else:
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
        msgpack = try_import_msgpack(error=True)
        policy_state['checkpoint_version'] = str(CHECKPOINT_VERSION)
        policy_state['policy_spec']['config'] = AlgorithmConfig._serialize_dict(policy_state['policy_spec']['config'])
        state_file = 'policy_state.msgpck'
        with open(os.path.join(export_dir, state_file), 'w+b') as f:
            msgpack.dump(policy_state, f)
    with open(os.path.join(export_dir, 'rllib_checkpoint.json'), 'w') as f:
        json.dump({'type': 'Policy', 'checkpoint_version': str(policy_state['checkpoint_version']), 'format': checkpoint_format, 'state_file': state_file, 'ray_version': ray.__version__, 'ray_commit': ray.__commit__}, f)
    if self.config['export_native_model_files']:
        self.export_model(os.path.join(export_dir, 'model'))