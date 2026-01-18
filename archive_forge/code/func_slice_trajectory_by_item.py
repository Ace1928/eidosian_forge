import os
from collections import defaultdict
import numpy as np
import minerl
import time
from typing import List
import collections
import os
import cv2
import gym
from minerl.data import DataPipeline
import sys
import time
from collections import deque, defaultdict
from enum import Enum
def slice_trajectory_by_item(self, trajectory):
    if trajectory is None:
        trajectory = TrajectoryDataPipeline.load_data(self.path_to_trajectory)
    state, action, reward, next_state, done = trajectory
    if self.length != len(reward):
        print(self.length, len(reward))
        raise NameError('Please, double check trajectory')
    result = defaultdict(list)
    for item in self.chain:
        if item.end - item.begin < 4:
            continue
        sliced_state = self.extract_from_dict(state, item.begin, item.end)
        sliced_action = self.extract_from_dict(action, item.begin, item.end)
        sliced_reward = reward[item.begin:item.end]
        sliced_next_state = self.extract_from_dict(next_state, item.begin, item.end)
        sliced_done = done[item.begin:item.end]
        result[item.name].append([sliced_state, sliced_action, sliced_reward, sliced_next_state, sliced_done])
    return result