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
def compute_q_values(policy: Policy, model: ModelV2, input_batch: SampleBatch, state_batches=None, seq_lens=None, explore=None, is_training: bool=False):
    config = policy.config
    model_out, state = model(input_batch, state_batches or [], seq_lens)
    if config['num_atoms'] > 1:
        action_scores, z, support_logits_per_action, logits, dist = model.get_q_value_distributions(model_out)
    else:
        action_scores, logits, dist = model.get_q_value_distributions(model_out)
    if config['dueling']:
        state_score = model.get_state_value(model_out)
        if config['num_atoms'] > 1:
            support_logits_per_action_mean = tf.reduce_mean(support_logits_per_action, 1)
            support_logits_per_action_centered = support_logits_per_action - tf.expand_dims(support_logits_per_action_mean, 1)
            support_logits_per_action = tf.expand_dims(state_score, 1) + support_logits_per_action_centered
            support_prob_per_action = tf.nn.softmax(logits=support_logits_per_action)
            value = tf.reduce_sum(input_tensor=z * support_prob_per_action, axis=-1)
            logits = support_logits_per_action
            dist = support_prob_per_action
        else:
            action_scores_mean = reduce_mean_ignore_inf(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            value = state_score + action_scores_centered
    else:
        value = action_scores
    return (value, logits, dist, state)