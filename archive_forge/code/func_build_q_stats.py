from typing import Dict
import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.algorithms.simple_q.utils import Q_SCOPE, Q_TARGET_SCOPE
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import get_categorical_class_with_temperature
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import LearningRateSchedule, TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration import ParameterNoise
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.tf_utils import (
from ray.rllib.utils.typing import AlgorithmConfigDict, ModelGradients, TensorType
def build_q_stats(policy: Policy, batch) -> Dict[str, TensorType]:
    return dict({'cur_lr': tf.cast(policy.cur_lr, tf.float64)}, **policy.q_loss.stats)