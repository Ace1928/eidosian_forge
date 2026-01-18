import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def get_cycle_path(self, curr_node, goal_node_index):
    for dep in curr_node['deps']:
        if dep == goal_node_index:
            return [curr_node['address']]
    for dep in curr_node['deps']:
        path = self.get_cycle_path(self.get_by_address(dep), goal_node_index)
        if len(path) > 0:
            path.insert(0, curr_node['address'])
            return path
    return []