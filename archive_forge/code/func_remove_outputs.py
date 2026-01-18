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
@register_strategy('Remove outputs')
def remove_outputs(cur_graph, cur_inps, granularity):
    granularity = max(1, granularity // 2)
    for idx, node in enumerate(cur_graph.nodes):
        node.idx = idx
        if node.op == 'output':
            output = node
            break
    if isinstance(output.args[0], fx.Node):
        return None
    output_args = sorted(output.args[0], key=lambda x: x.idx if isinstance(x, fx.Node) else int(1000000000.0))
    if len(output_args) == 1:
        return None
    for idx in range(0, len(output_args), granularity):
        output.args = (output_args[:idx] + output_args[idx + granularity:],)
        if graph_fails(cur_graph, cur_inps):
            return ReproState(cur_graph, cur_inps)
    return None