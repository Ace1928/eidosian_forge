from functools import partial
import numpy as np
import gymnasium as gym
import logging
import tree
from typing import Dict, List, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_tfp
from ray.rllib.utils.typing import (
def cql_loss(policy: Policy, model: ModelV2, dist_class: Type[TFActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    logger.info(f'Current iteration = {policy.cur_iter}')
    policy.cur_iter += 1
    deterministic = policy.config['_deterministic_loss']
    assert not deterministic
    twin_q = policy.config['twin_q']
    discount = policy.config['gamma']
    bc_iters = policy.config['bc_iters']
    cql_temp = policy.config['temperature']
    num_actions = policy.config['num_actions']
    min_q_weight = policy.config['min_q_weight']
    use_lagrange = policy.config['lagrangian']
    target_action_gap = policy.config['lagrangian_thresh']
    obs = train_batch[SampleBatch.CUR_OBS]
    actions = tf.cast(train_batch[SampleBatch.ACTIONS], tf.float32)
    rewards = tf.cast(train_batch[SampleBatch.REWARDS], tf.float32)
    next_obs = train_batch[SampleBatch.NEXT_OBS]
    terminals = train_batch[SampleBatch.TERMINATEDS]
    model_out_t, _ = model(SampleBatch(obs=obs, _is_training=True), [], None)
    model_out_tp1, _ = model(SampleBatch(obs=next_obs, _is_training=True), [], None)
    target_model_out_tp1, _ = policy.target_model(SampleBatch(obs=next_obs, _is_training=True), [], None)
    action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
    action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
    action_dist_t = action_dist_class(action_dist_inputs_t, model)
    policy_t, log_pis_t = action_dist_t.sample_logp()
    log_pis_t = tf.expand_dims(log_pis_t, -1)
    alpha_loss = -tf.reduce_mean(model.log_alpha * tf.stop_gradient(log_pis_t + model.target_entropy))
    alpha = tf.math.exp(model.log_alpha)
    if policy.cur_iter >= bc_iters:
        min_q, _ = model.get_q_values(model_out_t, policy_t)
        if twin_q:
            twin_q_, _ = model.get_twin_q_values(model_out_t, policy_t)
            min_q = tf.math.minimum(min_q, twin_q_)
        actor_loss = tf.reduce_mean(tf.stop_gradient(alpha) * log_pis_t - min_q)
    else:
        bc_logp = action_dist_t.logp(actions)
        actor_loss = tf.reduce_mean(tf.stop_gradient(alpha) * log_pis_t - bc_logp)
    action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
    action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
    policy_tp1, _ = action_dist_tp1.sample_logp()
    q_t, _ = model.get_q_values(model_out_t, actions)
    q_t_selected = tf.squeeze(q_t, axis=-1)
    if twin_q:
        twin_q_t, _ = model.get_twin_q_values(model_out_t, actions)
        twin_q_t_selected = tf.squeeze(twin_q_t, axis=-1)
    q_tp1, _ = policy.target_model.get_q_values(target_model_out_tp1, policy_tp1)
    if twin_q:
        twin_q_tp1, _ = policy.target_model.get_twin_q_values(target_model_out_tp1, policy_tp1)
        q_tp1 = tf.math.minimum(q_tp1, twin_q_tp1)
    q_tp1_best = tf.squeeze(input=q_tp1, axis=-1)
    q_tp1_best_masked = (1.0 - tf.cast(terminals, tf.float32)) * q_tp1_best
    q_t_target = tf.stop_gradient(rewards + discount ** policy.config['n_step'] * q_tp1_best_masked)
    base_td_error = tf.math.abs(q_t_selected - q_t_target)
    if twin_q:
        twin_td_error = tf.math.abs(twin_q_t_selected - q_t_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error
    critic_loss_1 = tf.keras.losses.MSE(q_t_selected, q_t_target)
    if twin_q:
        critic_loss_2 = tf.keras.losses.MSE(twin_q_t_selected, q_t_target)
    rand_actions, _ = policy._random_action_generator.get_exploration_action(action_distribution=action_dist_class(tf.tile(action_dist_tp1.inputs, (num_actions, 1)), model), timestep=0, explore=True)
    curr_actions, curr_logp = policy_actions_repeat(model, action_dist_class, model_out_t, num_actions)
    next_actions, next_logp = policy_actions_repeat(model, action_dist_class, model_out_tp1, num_actions)
    q1_rand = q_values_repeat(model, model_out_t, rand_actions)
    q1_curr_actions = q_values_repeat(model, model_out_t, curr_actions)
    q1_next_actions = q_values_repeat(model, model_out_t, next_actions)
    if twin_q:
        q2_rand = q_values_repeat(model, model_out_t, rand_actions, twin=True)
        q2_curr_actions = q_values_repeat(model, model_out_t, curr_actions, twin=True)
        q2_next_actions = q_values_repeat(model, model_out_t, next_actions, twin=True)
    random_density = np.log(0.5 ** int(curr_actions.shape[-1]))
    cat_q1 = tf.concat([q1_rand - random_density, q1_next_actions - tf.stop_gradient(next_logp), q1_curr_actions - tf.stop_gradient(curr_logp)], 1)
    if twin_q:
        cat_q2 = tf.concat([q2_rand - random_density, q2_next_actions - tf.stop_gradient(next_logp), q2_curr_actions - tf.stop_gradient(curr_logp)], 1)
    min_qf1_loss_ = tf.reduce_mean(tf.reduce_logsumexp(cat_q1 / cql_temp, axis=1)) * min_q_weight * cql_temp
    min_qf1_loss = min_qf1_loss_ - tf.reduce_mean(q_t) * min_q_weight
    if twin_q:
        min_qf2_loss_ = tf.reduce_mean(tf.reduce_logsumexp(cat_q2 / cql_temp, axis=1)) * min_q_weight * cql_temp
        min_qf2_loss = min_qf2_loss_ - tf.reduce_mean(twin_q_t) * min_q_weight
    if use_lagrange:
        alpha_prime = tf.clip_by_value(model.log_alpha_prime.exp(), 0.0, 1000000.0)[0]
        min_qf1_loss = alpha_prime * (min_qf1_loss - target_action_gap)
        if twin_q:
            min_qf2_loss = alpha_prime * (min_qf2_loss - target_action_gap)
            alpha_prime_loss = 0.5 * (-min_qf1_loss - min_qf2_loss)
        else:
            alpha_prime_loss = -min_qf1_loss
    cql_loss = [min_qf1_loss]
    if twin_q:
        cql_loss.append(min_qf2_loss)
    critic_loss = [critic_loss_1 + min_qf1_loss]
    if twin_q:
        critic_loss.append(critic_loss_2 + min_qf2_loss)
    policy.q_t = q_t_selected
    policy.policy_t = policy_t
    policy.log_pis_t = log_pis_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.log_alpha_value = model.log_alpha
    policy.alpha_value = alpha
    policy.target_entropy = model.target_entropy
    policy.cql_loss = cql_loss
    if use_lagrange:
        policy.log_alpha_prime_value = model.log_alpha_prime[0]
        policy.alpha_prime_value = alpha_prime
        policy.alpha_prime_loss = alpha_prime_loss
    if use_lagrange:
        return actor_loss + tf.math.add_n(critic_loss) + alpha_loss + alpha_prime_loss
    return actor_loss + tf.math.add_n(critic_loss) + alpha_loss