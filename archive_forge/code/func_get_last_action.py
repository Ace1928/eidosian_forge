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
def get_last_action(self) -> Action:
    actions = self.__getitem__('actions')
    if actions:
        return actions[-1]
    else:
        return None