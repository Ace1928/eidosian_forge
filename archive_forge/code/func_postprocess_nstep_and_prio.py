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
def postprocess_nstep_and_prio(policy: Policy, batch: SampleBatch, other_agent=None, episode=None) -> SampleBatch:
    if policy.config['n_step'] > 1:
        adjust_nstep(policy.config['n_step'], policy.config['gamma'], batch)
    if PRIO_WEIGHTS not in batch:
        batch[PRIO_WEIGHTS] = np.ones_like(batch[SampleBatch.REWARDS])
    if batch.count > 0 and policy.config['replay_buffer_config'].get('worker_side_prioritization', False):
        td_errors = policy.compute_td_error(batch[SampleBatch.OBS], batch[SampleBatch.ACTIONS], batch[SampleBatch.REWARDS], batch[SampleBatch.NEXT_OBS], batch[SampleBatch.TERMINATEDS], batch[PRIO_WEIGHTS])
        epsilon = policy.config.get('replay_buffer_config', {}).get('prioritized_replay_eps') or policy.config.get('prioritized_replay_eps')
        if epsilon is None:
            raise ValueError('prioritized_replay_eps not defined in config.')
        new_priorities = np.abs(convert_to_numpy(td_errors)) + epsilon
        batch[PRIO_WEIGHTS] = new_priorities
    return batch