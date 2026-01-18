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
def generate_final_chain(envs=('MineRLObtainIronPickaxe-v0',), data_dir='../data/'):
    """
    generates final chain
    it may sampled randomly, but be careful short chains give poor results
    :param envs: number of envs
    :param data_dir:
    :return:
    """
    return generate_best_chains(envs=envs, data_dir=data_dir)[-1]