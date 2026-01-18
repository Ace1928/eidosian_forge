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
def additional_logs():
    tf.summary.scalar('learning_rate', learning_rate_fn(iterations))
    n_episodes = info_queue.size()
    log_episode_frequency = 1
    n_episodes -= n_episodes % log_episode_frequency
    if tf.not_equal(n_episodes, 0):
        episode_stats = info_queue.dequeue_many(n_episodes)
        episode_keys = ['episode_num_frames', 'episode_return', 'episode_raw_return']
        for key, values in zip(episode_keys, episode_stats):
            for value in tf.split(values, values.shape[0] // log_episode_frequency):
                tf.summary.scalar(key, tf.reduce_mean(value))
        for frames, ep_return, raw_return in zip(*episode_stats):
            logging.info('Return: %f Raw return: %f Frames: %i', ep_return, raw_return, frames)