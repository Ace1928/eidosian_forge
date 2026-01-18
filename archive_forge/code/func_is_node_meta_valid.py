import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def is_node_meta_valid(node: Optional[torch.fx.Node]):
    if node is None:
        return True
    if 'example_value' not in node.meta:
        return False
    return True