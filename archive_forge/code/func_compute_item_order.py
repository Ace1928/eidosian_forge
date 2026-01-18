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
@classmethod
def compute_item_order(cls, trajectory, return_time_indexes=False):
    return cls.to_old_chain_format(cls.extract_subtasks(trajectory), return_time_indexes=return_time_indexes)