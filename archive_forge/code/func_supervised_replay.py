import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import glob
import random
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import gym
import minerl
import os
import gym
import minerl
import numpy as np
from absl import flags
import argparse
import network
from utils import TrajectoryInformation, DummyDataLoader, TrajectoryDataPipeline
from collections import deque, defaultdict
import numpy as np
@tf.function
def supervised_replay(replay_obs_list, replay_act_list, memory_state, carry_state):
    replay_obs_array = tf.concat(replay_obs_list, 0)
    replay_act_array = tf.concat(replay_act_list, 0)
    replay_memory_state_array = tf.concat(memory_state, 0)
    replay_carry_state_array = tf.concat(carry_state, 0)
    memory_state = replay_memory_state_array
    carry_state = replay_carry_state_array
    batch_size = replay_obs_array.shape[0]
    tf.print('batch_size: ', batch_size)
    with tf.GradientTape() as tape:
        act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(0, batch_size):
            prediction = model(tf.expand_dims(replay_obs_array[i, :, :, :], 0), memory_state, carry_state, training=True)
            act_pi = prediction[0]
            memory_state = prediction[2]
            carry_state = prediction[3]
            act_probs = act_probs.write(i, act_pi[0])
        act_probs = act_probs.stack()
        tf.print('replay_act_array: ', replay_act_array)
        tf.print('tf.argmax(act_probs, 1): ', tf.argmax(act_probs, 1))
        replay_act_array_onehot = tf.one_hot(replay_act_array, num_actions)
        replay_act_array_onehot = tf.reshape(replay_act_array_onehot, (batch_size, num_actions))
        act_loss = cce_loss_logits(replay_act_array_onehot, act_probs)
        regularization_loss = tf.reduce_sum(model.losses)
        total_loss = act_loss + 1e-05 * regularization_loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return (total_loss, memory_state, carry_state)