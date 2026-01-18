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
def get_colored_vertexes(chain):
    """
        determines the color for each vertex
        item + its actions have the same color
        :param chain:
        :return: {vertex: color}
        """
    vertexes = VisTools.get_all_vertexes_from_edges(chain)
    result = {}
    colors = ['#ffe6cc', '#ccffe6']
    current_color = 0
    for vertex in vertexes:
        result[vertex] = colors[current_color]
        bool_ = True
        for action in ['equip', 'craft', 'nearbyCraft', 'nearbySmelt', 'place']:
            if action + ':' in vertex:
                bool_ = False
        if bool_:
            current_color = (current_color + 1) % len(colors)
    return result