import torch.fx as fx
import copy
import torch
import math
import sys
from typing import Callable, List
from functools import wraps, partial
from dataclasses import dataclass
from .compile_utils import get_placeholders, get_outputs
from torch.utils._content_store import ContentStoreWriter
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
import os
@register_strategy('Eliminate dead code')
def eliminate_dead_code(cur_graph, cur_inps, granularity):
    if cur_graph.eliminate_dead_code() and graph_fails(cur_graph, cur_inps):
        return ReproState(cur_graph, cur_inps)
    return None