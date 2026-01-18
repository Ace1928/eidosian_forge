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
def maybe_add_time_dimension(self, input_dict: Dict[str, TensorType], seq_lens: TensorType, framework: str=None):
    """Adds a time dimension for recurrent RLModules.

        Args:
            input_dict: The input dict.
            seq_lens: The sequence lengths.
            framework: The framework to use for adding the time dimensions.
                If None, will default to the framework of the policy.

        Returns:
            The input dict, with a possibly added time dimension.
        """
    if self.config.get('_enable_new_api_stack', False) and hasattr(self, 'model') and self.model.is_stateful():
        ret = {}
        framework = framework or self.model.framework

        def _add_time_dimension(inputs):
            inputs = add_time_dimension(inputs, seq_lens=seq_lens, framework=framework, time_major=self.config.get('model', {}).get('_time_major', False))
            return inputs

        def _add_state_out_time_dimension(inputs):
            v_w_two_time_dims = _add_time_dimension(inputs)
            if framework == 'tf2':
                return tf.squeeze(v_w_two_time_dims, axis=2)
            elif framework == 'torch':
                return torch.squeeze(v_w_two_time_dims, axis=2)
            elif framework == 'np':
                shape = v_w_two_time_dims.shape
                padded_batch_dim = shape[0]
                padded_time_dim = shape[1]
                other_dims = shape[3:]
                new_shape = (padded_batch_dim, padded_time_dim) + other_dims
                return v_w_two_time_dims.reshape(new_shape)
            else:
                raise ValueError(f'Framework {framework} not implemented!')
        for k, v in input_dict.items():
            if k == SampleBatch.INFOS:
                ret[k] = _add_time_dimension(v)
            elif k == SampleBatch.SEQ_LENS:
                ret[k] = v
            elif k == STATE_IN:
                assert self.view_requirements[k].batch_repeat_value != 1
                ret[k] = v
            elif k == STATE_OUT:
                assert self.view_requirements[k].batch_repeat_value == 1
                ret[k] = tree.map_structure(_add_state_out_time_dimension, v)
            else:
                ret[k] = tree.map_structure(_add_time_dimension, v)
        return SampleBatch(ret)
    else:
        return input_dict