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
def apply_checkpoint_execution_state_in_allocator(self):
    """
        Checkpoint the current execution state in the caching allocator so that
        additional cudagraph recordings can be made respecting existent live storages.
        """
    self.debug_checkpointing_counter += 1
    log.debug('Checkpointing cuda caching allocator state. Number of checkpoints %d', self.debug_checkpointing_counter)
    state = self.current_node.checkpointed_caching_state
    device = self.current_node.device
    assert state is not None and device is not None
    stale_storages: List[int] = []
    self.current_node.remove_path_cached_tensors()
    live_storages_wrappers = list(self.current_node.path_live_weakrefs())
    live_storages_weak_refs = [t() for t in live_storages_wrappers]
    ptrs_to_deallocate = self.current_node.data_ptrs_dead_since_invocation()
    torch._C._cuda_setCheckpointPoolState(device, state, stale_storages, live_storages_weak_refs)
    for ptr in set(ptrs_to_deallocate):
        torch._C._cuda_cudaCachingAllocator_raw_delete(ptr)
    if config.triton.slow_path_cudagraph_asserts:
        check_memory_pool(self.device_index, self.cuda_graphs_thread_pool, live_storages_wrappers)
        for wrapper in live_storages_wrappers:
            assert wrapper()
            assert torch._C._has_Standard_Deleter(wrapper())
            assert wrapper.data_ptr() not in ptrs_to_deallocate