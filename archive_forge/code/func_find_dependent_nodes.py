import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def find_dependent_nodes(src_node, cur_node):
    for input_node in cur_node.all_input_nodes:
        if input_node in node_list:
            dep_set.add(input_node)
        if input_node not in visited_node_set:
            visited_node_set.add(input_node)
            find_dependent_nodes(src_node, input_node)