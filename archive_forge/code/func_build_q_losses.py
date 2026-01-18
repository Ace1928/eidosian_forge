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
def build_q_losses(policy: Policy, model, _, train_batch: SampleBatch) -> TensorType:
    """Constructs the loss for DQNTFPolicy.

    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        train_batch: The training data.

    Returns:
        TensorType: A single loss tensor.
    """
    config = policy.config
    q_t, q_logits_t, q_dist_t, _ = compute_q_values(policy, model, SampleBatch({'obs': train_batch[SampleBatch.CUR_OBS]}), state_batches=None, explore=False)
    q_tp1, q_logits_tp1, q_dist_tp1, _ = compute_q_values(policy, policy.target_model, SampleBatch({'obs': train_batch[SampleBatch.NEXT_OBS]}), state_batches=None, explore=False)
    if not hasattr(policy, 'target_q_func_vars'):
        policy.target_q_func_vars = policy.target_model.variables()
    one_hot_selection = tf.one_hot(tf.cast(train_batch[SampleBatch.ACTIONS], tf.int32), policy.action_space.n)
    q_t_selected = tf.reduce_sum(q_t * one_hot_selection, 1)
    q_logits_t_selected = tf.reduce_sum(q_logits_t * tf.expand_dims(one_hot_selection, -1), 1)
    if config['double_q']:
        q_tp1_using_online_net, q_logits_tp1_using_online_net, q_dist_tp1_using_online_net, _ = compute_q_values(policy, model, SampleBatch({'obs': train_batch[SampleBatch.NEXT_OBS]}), state_batches=None, explore=False)
        q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
        q_tp1_best_one_hot_selection = tf.one_hot(q_tp1_best_using_online_net, policy.action_space.n)
        q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_dist_tp1_best = tf.reduce_sum(q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1)
    else:
        q_tp1_best_one_hot_selection = tf.one_hot(tf.argmax(q_tp1, 1), policy.action_space.n)
        q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_dist_tp1_best = tf.reduce_sum(q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1)
    loss_fn = huber_loss if policy.config['td_error_loss_fn'] == 'huber' else l2_loss
    policy.q_loss = QLoss(q_t_selected, q_logits_t_selected, q_tp1_best, q_dist_tp1_best, train_batch[PRIO_WEIGHTS], tf.cast(train_batch[SampleBatch.REWARDS], tf.float32), tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32), config['gamma'], config['n_step'], config['num_atoms'], config['v_min'], config['v_max'], loss_fn)
    return policy.q_loss.loss