import gymnasium as gym
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, TYPE_CHECKING
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.policy import eager_tf_policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils import add_mixins, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import (
@override(TFPolicy)
def extra_compute_action_fetches(self):
    return dict(base.extra_compute_action_fetches(self), **self._extra_action_fetches)