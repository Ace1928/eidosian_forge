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
def save_chain_in_graph(chain_to_save, name='out', format_='png'):
    """
        saving image of a graph using draw_graph method
        :param chain_to_save:
        :param name: filename
        :param format_: file type e.g. ".png" or ".svg"
        :return:
        """
    graph = []
    for c_index, item in enumerate(chain_to_save):
        if c_index:
            graph.append([str(c_index) + '\n' + VisTools.replace_with_name(chain_to_save[c_index - 1]), str(c_index + 1) + '\n' + VisTools.replace_with_name(item)])
    VisTools.draw_graph(name, graph=graph, format_=format_, vertex_colors=VisTools.get_colored_vertexes(graph))