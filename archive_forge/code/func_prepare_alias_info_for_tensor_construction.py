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
def prepare_alias_info_for_tensor_construction(self, out_alias_info: Optional[OutputAliasInfo], metadata: Union[Dict[str, Any], int, None]) -> Union[UntypedStorage, None, int]:
    if isinstance(metadata, (int, type(None))) or out_alias_info is UnaliasedStorage:
        return None
    if isinstance(out_alias_info, AliasesPriorGraphOutput):
        depth, existing_output_index = out_alias_info.index
        ref = self.path_weakrefs[depth][existing_output_index]
        assert ref is not None
        return torch.UntypedStorage._new_with_weak_ptr(ref())
    assert isinstance(out_alias_info, AliasesNewOutput)
    return out_alias_info.index