import json
import math
import os
import re
from typing import Dict, List, Optional, Set
import torch
import torch.utils.benchmark as benchmark
from torch._C._profiler import (
from torch.profiler import profile
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
def same_ops(list1, list2):
    if len(list1) != len(list2):
        return False
    for op1, op2 in zip(list1, list2):
        if op1.name != op2.name:
            return False
    return True