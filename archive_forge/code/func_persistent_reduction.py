import builtins
import copy
import functools
import hashlib
import inspect
import json
import logging
import math
import operator
import os
import os.path
import re
import threading
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import torch
import torch.autograd.profiler as autograd_profiler
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import dynamo_timed
from torch.utils._triton import has_triton, has_triton_package
from . import config
from .codecache import cache_dir, CudaKernelParamCache
from .coordinate_descent_tuner import CoordescTuner
from .ir import ReductionHint, TileHint
from .utils import (
def persistent_reduction(size_hints, reduction_hint=False, triton_meta=None, filename=None, inductor_meta=None):
    xnumel, rnumel = size_hints
    configs = [triton_config_reduction(size_hints, xblock, rnumel) for xblock in (1, 8, 32, 128) if rnumel * xblock <= 4096 and xblock <= xnumel]
    if reduction_hint == ReductionHint.INNER and rnumel >= 256:
        configs = configs[:1]
    elif reduction_hint == ReductionHint.OUTER:
        configs = configs[-1:]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        configs = [triton_config_reduction(size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, rnumel)]
    for c in configs:
        c.kwargs.pop('RBLOCK')
    if disable_pointwise_autotuning():
        configs = configs[:1]
    return cached_autotune(size_hints, configs, triton_meta=triton_meta, inductor_meta=inductor_meta, filename=filename, heuristic_type=HeuristicType.PERSISTENT_REDUCTION)