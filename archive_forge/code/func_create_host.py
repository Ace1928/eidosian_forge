import collections
import math
import os
import time
import observation
import networks
import gym
from gym.spaces import Dict, Discrete, Box, Tuple
import argparse
from absl import flags
from absl import logging
import grpc
import utils
import vtrace
from parametric_distribution import get_parametric_distribution_for_action_space
import tensorflow as tf
def create_host(i, host, inference_devices):
    with tf.device(host):
        server_address = 'localhost:8686'
        server = grpc.Server([server_address])
        store = utils.UnrollStore(num_envs, unroll_length, (action_specs, env_output_specs, agent_output_specs))
        env_run_ids = utils.Aggregator(num_envs, tf.TensorSpec([], tf.int64, 'run_ids'))
        env_infos = utils.Aggregator(num_envs, info_specs, 'env_infos')
        first_agent_states = utils.Aggregator(num_envs, agent_state_specs, 'first_agent_states')
        agent_states = utils.Aggregator(num_envs, agent_state_specs, 'agent_states')
        actions = utils.Aggregator(num_envs, action_specs, 'actions')
        unroll_specs[0] = Unroll(agent_state_specs, *store.unroll_specs)
        unroll_queue = utils.StructuredFIFOQueue(1, unroll_specs[0])

        def add_batch_size(ts):
            return tf.TensorSpec([inference_batch_size] + list(ts.shape), ts.dtype, ts.name)
        inference_specs = (tf.TensorSpec([], tf.int32, 'env_id'), tf.TensorSpec([], tf.int64, 'run_id'), env_output_specs, tf.TensorSpec([], tf.float32, 'raw_reward'))
        inference_specs = tf.nest.map_structure(add_batch_size, inference_specs)

        def create_inference_fn(inference_device):

            @tf.function(input_signature=inference_specs)
            def inference(env_ids, run_ids, env_outputs, raw_rewards):
                previous_run_ids = env_run_ids.read(env_ids)
                env_run_ids.replace(env_ids, run_ids)
                reset_indices = tf.where(tf.not_equal(previous_run_ids, run_ids))[:, 0]
                envs_needing_reset = tf.gather(env_ids, reset_indices)
                if tf.not_equal(tf.shape(envs_needing_reset)[0], 0):
                    tf.print('Environment ids needing reset:', envs_needing_reset)
                store.reset(envs_needing_reset)
                env_infos.reset(envs_needing_reset)
                initial_agent_states = agent.initial_state(tf.shape(envs_needing_reset)[0])
                first_agent_states.replace(envs_needing_reset, initial_agent_states)
                agent_states.replace(envs_needing_reset, initial_agent_states)
                actions.reset(envs_needing_reset)
                env_infos.add(env_ids, (0, env_outputs.reward, raw_rewards))
                done_ids = tf.gather(env_ids, tf.where(env_outputs.done)[:, 0])
                if i == 0:
                    info_queue.enqueue_many(env_infos.read(done_ids))
                env_infos.reset(done_ids)
                env_infos.add(env_ids, (num_action_repeats, 0.0, 0.0))
                prev_actions = actions.read(env_ids)
                input_ = encode((prev_actions, env_outputs))
                prev_agent_states = agent_states.read(env_ids)
                with tf.device(inference_device):

                    @tf.function
                    def agent_inference(*args):
                        return agent(*decode(args), is_training=False)
                    agent_outputs, curr_agent_states = agent_inference(*input_, prev_agent_states)
                completed_ids, unrolls = store.append(env_ids, (prev_actions, env_outputs, agent_outputs))
                unrolls = Unroll(first_agent_states.read(completed_ids), *unrolls)
                unroll_queue.enqueue_many(unrolls)
                first_agent_states.replace(completed_ids, agent_states.read(completed_ids))
                agent_states.replace(env_ids, curr_agent_states)
                actions.replace(env_ids, agent_outputs.action)
                return agent_outputs.action
            return inference
        with strategy.scope():
            server.bind([create_inference_fn(d) for d in inference_devices])
        server.start()
        unroll_queues.append(unroll_queue)
        servers.append(server)