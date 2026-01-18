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
def draw_graph(file_name, graph, format_='svg', vertex_colors=None):
    """
        drawing png graph from the list of edges
        :param vertex_colors:
        :param format_: resulted file format
        :param file_name: file_name
        :param graph: graph file with format: (left_edge, right_edge) or (left_edge, right_edge, label)
        :return: None
        """
    import pygraphviz as pgv
    g_out = pgv.AGraph(strict=False, directed=True)
    for i in graph:
        g_out.add_edge(i[0], i[1], color='black')
        edge = g_out.get_edge(i[0], i[1])
        if len(i) > 2:
            edge.attr['label'] = i[2]
    g_out.node_attr['style'] = 'filled'
    if vertex_colors:
        for vertex, color in vertex_colors.items():
            g_out.get_node(vertex).attr['fillcolor'] = color
    g_out.layout(prog='dot')
    g_out.draw(path='{file_name}.{format_}'.format(**locals()))