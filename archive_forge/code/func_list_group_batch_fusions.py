import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def list_group_batch_fusions(pre_grad=True) -> List[str]:
    if pre_grad:
        return list(PRE_GRAD_FUSIONS.keys())
    else:
        return list(POST_GRAD_FUSIONS.keys())