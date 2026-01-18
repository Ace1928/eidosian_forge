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
def dealloc_current_path_weakrefs(self):
    for node in self.current_node._path_from_root:
        assert len(node.tensor_weakrefs) == len(node.stack_traces)
        for t, stack_trace in zip(node.tensor_weakrefs, node.stack_traces):
            ten = None if t is None else t()
            if ten is None:
                continue
            stack_trace = stack_trace.strip() if stack_trace else '[Could not find stack trace]'
            msg = f'Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: {stack_trace}. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.'
            torch._C._set_storage_access_error_msg(ten, msg)
    deleted = set()
    for storage_ref in self.current_node.path_live_weakrefs():
        if storage_ref() and storage_ref.data_ptr() not in deleted:
            deleted.add(storage_ref.data_ptr())
            torch._C._free_And_Remove_DeleterFn(storage_ref())