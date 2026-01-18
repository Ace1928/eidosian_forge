import logging
from typing import Any, Dict, List, Tuple, Type, Union
from ray.rllib.algorithms.simple_q.utils import make_q_models
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import TargetNetworkMixin, LearningRateSchedule
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import concat_multi_gpu_td_errors, huber_loss
from ray.rllib.utils.typing import TensorStructType, TensorType
@override(TorchPolicyV2)
def extra_compute_grad_fetches(self) -> Dict[str, Any]:
    fetches = convert_to_numpy(concat_multi_gpu_td_errors(self))
    return dict({LEARNER_STATS_KEY: {}}, **fetches)