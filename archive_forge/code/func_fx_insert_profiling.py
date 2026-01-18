import dataclasses
import os
from typing import Any, List
import torch
from .utils import print_once
def fx_insert_profiling(gm: torch.fx.GraphModule, example_inputs: List[Any]):

    def _wrapped(*args):
        with torch.profiler.record_function('TORCHDYNAMO'):
            return gm.forward(*args)
    Profiler.unique_graphs += 1
    return _wrapped