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
def get_non_cudagraph_inps():
    non_cudagraph_inps = set()
    for t in itertools.chain(new_inputs, self.wrapped_function.constants):
        if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in existing_path_data_ptrs:
            non_cudagraph_inps.add(t.untyped_storage().data_ptr())
    return non_cudagraph_inps