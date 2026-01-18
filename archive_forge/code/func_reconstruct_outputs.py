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
def reconstruct_outputs(self):
    """Reconstruct output tensors according to their saved metadata and alias information"""
    if not self.cached_tensor_outputs:
        self._initialize_cached_tensors()
    outputs: List[Optional[Union[int, torch.Tensor]]] = []
    for i, (storage_info, metadata) in enumerate(zip(self.output_storage_alias, self.outputs_metadata)):
        if not isinstance(metadata, dict):
            assert isinstance(metadata, (int, type(None)))
            outputs.append(metadata)
            continue
        cached_t = self.cached_tensor_outputs[i]
        if cached_t is not None:
            outputs.append(cached_t)
            continue
        static_t = self.static_output_tensors[i]
        if static_t is not None:
            assert self.outputs_weakrefs[i] is None
            outputs.append(static_t)
            continue
        storage = self.prepare_alias_info_for_tensor_construction(storage_info, metadata)
        if isinstance(storage, UntypedStorage) or storage is None:
            out = self._reconstruct_from_tensor_metadata(metadata, storage)
        else:
            assert isinstance(storage, int)
            out = self._reconstruct_from_tensor_metadata(metadata, cast(torch.Tensor, outputs[storage]).untyped_storage())
        outputs.append(out)
        w = self.outputs_weakrefs[i]
        assert w is not None
        w.swap_weakref(out.untyped_storage()._weak_ref())
    return outputs