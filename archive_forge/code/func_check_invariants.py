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
def check_invariants(self, inputs: List[Tensor]) -> bool:
    """
        Checks if this node can be run. The same pattern of tensor liveness and tensors
        managed in the cudagraph private pool must remain stable.
        """
    for idx in self.cudagraph_managed_idxs:
        if inputs[idx].data_ptr() != self.static_input_data_ptrs[idx]:
            return False
    if not self._check_liveness(self.expected_dead_indices_before_graph, self.path_weakrefs):
        return False
    for idx in self.cudagraph_managed_idxs:
        inputs[idx] = None
    torch._check(self._check_liveness(self.expected_dead_indices_after_graph, self.path_weakrefs), lambda: 'TODO: graph recording observed an input tensor deallocate during graph  recording that did not occur during replay. Please file an issue.')
    return True