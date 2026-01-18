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
def cudagraphify_impl(model, inputs, static_input_idxs, *args, **kwargs):
    fn_cache: Dict[Tuple[int, ...], Callable[..., Any]] = {}
    int_key = [i for i, v in enumerate(inputs) if isinstance(v, int)]
    get_ints: Any = operator.itemgetter(*int_key) if int_key else lambda _: None
    del inputs

    def deferred_cudagraphify(inputs):
        int_key = get_ints(inputs)
        fn = fn_cache.get(int_key)
        if fn is not None:
            return fn(inputs)
        log.info('recording cudagraph tree for %s', int_key)
        check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
        new_static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
        copy_misaligned_inputs(inputs, check_input_idxs)
        fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)
        fn = align_inputs_from_check_idxs(fn, inputs_to_check=check_input_idxs)
        fn_cache[int_key] = fn
        return out
    return deferred_cudagraphify