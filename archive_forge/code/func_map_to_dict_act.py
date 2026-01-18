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
@staticmethod
def map_to_dict_act(handler_list: list, target_space: gym.spaces.space, ignore_keys=()):

    def _map_to_dict(i: int, src: list, key: str, gym_space: gym.spaces.space, dst: dict):
        if isinstance(gym_space, gym.spaces.Dict):
            dont_count = False
            inner_dict = collections.OrderedDict()
            for idx, (k, s) in enumerate(gym_space.spaces.items()):
                if key in ['equipped_items', 'mainhand']:
                    dont_count = True
                    i = _map_to_dict(i, src, k, s, inner_dict)
                else:
                    _map_to_dict(idx, src[i].T, k, s, inner_dict)
            dst[key] = inner_dict
            if dont_count:
                return i
            else:
                return i + 1
        else:
            dst[key] = src[i]
            return i + 1
    result = collections.OrderedDict()
    index = 0
    "\n        actions: ['action$forward', 'action$left', 'action$back', 'action$right', 'action$jump', 'action$sneak', 'action$sprint', \n        'action$attack', 'action$camera', 'action$place', 'action$equip', 'action$craft', 'action$nearbyCraft', 'action$nearbySmelt']\n\n        target_space.spaces.items(): odict_items([('attack', Discrete(2)), ('back', Discrete(2)), ('camera', Box(low=-180.0, high=180.0, shape=(2,))), \n        ('craft', Discrete(5)), ('equip', Discrete(8)), ('forward', Discrete(2)), ('jump', Discrete(2)), ('left', Discrete(2)), ('nearbyCraft', Discrete(8)), \n        ('nearbySmelt', Discrete(3)), ('place', Discrete(7)), ('right', Discrete(2)), ('sneak', Discrete(2)), ('sprint', Discrete(2))])\n        "
    key_list = ['forward', 'left', 'back', 'right', 'jump', 'sneak', 'sprint', 'attack', 'camera', 'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt']
    key_index = 0
    for key, space in target_space.spaces.items():
        key = key_list[key_index]
        key_index += 1
        if key in ignore_keys:
            continue
        index = _map_to_dict(index, handler_list, key, space, result)
    return result