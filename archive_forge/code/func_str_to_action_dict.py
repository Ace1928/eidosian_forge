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
def str_to_action_dict(action_):
    """
    str -> dict
    :param action_:
    :return:
    """
    a_, _, value = action_.split(':')
    if a_ == 'craft':
        value = craft[value].value
    elif a_ == 'equip':
        value = equip[value].value
    elif a_ == 'nearbyCraft':
        value = nearbyCraft[value].value
    elif a_ == 'nearbySmelt':
        value = nearbySmelt[value].value
    elif a_ == 'place':
        value = place[value].value
    return {a_: int(value)}