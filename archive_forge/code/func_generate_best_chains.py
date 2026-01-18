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
def generate_best_chains(envs=('MineRLObtainIronPickaxe-v0',), data_dir='../data/'):
    """
    generates final chain
    it may sampled randomly, but be careful short chains give poor results
    :param envs: number of envs
    :param data_dir:
    :return:
    """
    chains = all_chains_info(envs=envs, data_dir=data_dir)
    filtered = [c for c in chains if c.reward == max([_.reward for _ in chains])]
    filtered = [c for c in sorted(filtered, key=lambda x: x.length)][:60]
    filtered = [c for c in sorted(filtered, key=lambda x: len(x.chain)) if 25 < len(c.chain) <= 31]
    filtered_chains = []
    for chain in filtered:
        filtered_chains.append(chain.chain)
    return filtered_chains