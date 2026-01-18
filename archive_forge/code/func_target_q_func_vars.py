import logging
from typing import Dict, List
import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.eager_tf_policy import EagerTFPolicy
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.utils.tf_utils import make_tf_callable
from ray.rllib.utils.typing import (
@property
def target_q_func_vars(self):
    if not hasattr(self, '_target_q_func_vars'):
        if self.config.get('_enable_new_api_stack', False):
            self._target_q_func_vars = self.target_model.variables
        else:
            self._target_q_func_vars = self.target_model.variables()
    return self._target_q_func_vars