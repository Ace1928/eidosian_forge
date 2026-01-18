important optimization when chaining multiple CUDA graphs together, as it
from __future__ import annotations
import contextlib
import dataclasses
import functools
import gc
import itertools
import logging
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from enum import auto, Enum
from typing import (
import torch.fx
from torch import Tensor
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.compile_fx import (
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.types import _bool
from torch.utils import _pytree as pytree
from torch.utils.weak import TensorWeakRef
from . import config
def check_warn_on_unable_to_start_executing(self, function_id: FunctionID):
    """Warn if we in a potential loop where we are unable to hit fast path"""
    if function_id in self.warned_functions or not self.in_new_torch_compile_invocation():
        return
    existing_nodes = [node for node in self.current_node._path_from_root if node.wrapped_function.id == function_id]
    if len(existing_nodes) <= 1:
        return
    parents = {n.parent.wrapped_function.id for n in itertools.chain(existing_nodes, (self.current_node,)) if n.parent is not None}
    if len(parents) == len(existing_nodes):
        return
    self.warned_functions.add(function_id)
    warnings.warn('Unable to hit fast path of CUDAGraphs because of pending, uninvoked backwards. Consider running with torch.no_grad() or using torch.compiler.cudagraph_mark_step_begin() before each model invocation')