import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
from ray.rllib.algorithms.sac.sac_tf_model import SACTFModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.tf_utils import huber_loss, make_tf_callable
from ray.rllib.utils.typing import (
def get_distribution_inputs_and_class(policy: Policy, model: ModelV2, obs_batch: TensorType, *, explore: bool=True, **kwargs) -> Tuple[TensorType, Type[TFActionDistribution], List[TensorType]]:
    """The action distribution function to be used the algorithm.

    An action distribution function is used to customize the choice of action
    distribution class and the resulting action distribution inputs (to
    parameterize the distribution object).
    After parameterizing the distribution, a `sample()` call
    will be made on it to generate actions.

    Args:
        policy: The Policy being queried for actions and calling this
            function.
        model: The SAC specific Model to use to generate the
            distribution inputs (see sac_tf|torch_model.py). Must support the
            `get_action_model_outputs` method.
        obs_batch: The observations to be used as inputs to the
            model.
        explore: Whether to activate exploration or not.

    Returns:
        Tuple[TensorType, Type[TFActionDistribution], List[TensorType]]: The
            dist inputs, dist class, and a list of internal state outputs
            (in the RNN case).
    """
    forward_out, state_out = model(SampleBatch(obs=obs_batch, _is_training=policy._get_is_training_placeholder()), [], None)
    distribution_inputs, _ = model.get_action_model_outputs(forward_out)
    action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
    return (distribution_inputs, action_dist_class, state_out)