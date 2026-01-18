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
def supervised_train(dataset, training_episode):
    for batch in dataset:
        episode_size = batch[0].shape[1]
        print('episode_size: ', episode_size)
        replay_obs_list = batch[0][0]
        replay_act_list = batch[1][0]
        memory_state = np.zeros([1, 128], dtype=np.float32)
        carry_state = np.zeros([1, 128], dtype=np.float32)
        step_length = 32
        total_loss = 0
        for episode_index in range(0, episode_size, step_length):
            obs = replay_obs_list[episode_index:episode_index + step_length, :, :, :]
            act = replay_act_list[episode_index:episode_index + step_length, :]
            if len(obs) != step_length:
                break
            total_loss, next_memory_state, next_carry_state = supervised_replay(obs, act, memory_state, carry_state)
            memory_state = next_memory_state
            carry_state = next_carry_state
            print('total_loss: ', total_loss)
            print('')
        with writer.as_default():
            tf.summary.scalar('total_loss', total_loss, step=training_episode)
            writer.flush()
        if training_episode % 100 == 0:
            model.save_weights(workspace_path + '/model/tree_supervised_model_' + str(training_episode))