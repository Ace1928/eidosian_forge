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
def all_chains_info(envs, data_dir):
    chains = []

    def get_reward(trajectory_):
        return int(sum(trajectory_[2]))
    for env_name in envs:
        data = minerl.data.make(env_name, data_dir=data_dir)
        for index, trajectory_name in enumerate(sorted(data.get_trajectory_names())):
            print(trajectory_name)
            trajectory = TrajectoryDataPipeline.load_data_no_pov(os.path.join(data_dir, env_name, trajectory_name))
            chain, time_indexes = TrajectoryInformation.compute_item_order(trajectory, return_time_indexes=True)
            chains.append(ChainInfo(chain=chain, reward=get_reward(trajectory), env_name=env_name, trajectory_name=trajectory_name, id_=index, length=len(trajectory[2]), time_indexes=time_indexes))
    return chains