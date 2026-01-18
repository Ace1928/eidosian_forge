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
def graph_fails(graph, inps):
    nonlocal num_queries
    graph = copy.deepcopy(graph)
    num_queries += 1
    mod = fx.GraphModule(fail_f, graph)
    mod.graph.lint()
    return module_fails(mod, inps)