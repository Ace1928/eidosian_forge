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
def load_cached_autotuning(cache_filename: str, configs_hash: str, configs: List[Config]):
    """
    Read a cached autotuning result from disk
    """
    if not os.path.exists(cache_filename):
        return None
    with open(cache_filename) as fd:
        best_config = json.loads(fd.read())
    if best_config.pop('configs_hash', None) != configs_hash:
        return None
    if config.coordinate_descent_tuning and best_config.pop('found_by_coordesc', False):
        num_warps = best_config.pop('num_warps')
        num_stages = best_config.pop('num_stages')
        triton_config = Config(best_config, num_warps=num_warps, num_stages=num_stages)
        triton_config.found_by_coordesc = True
        return triton_config
    matching_configs = [cfg for cfg in configs if all((val == best_config.get(key) for key, val in cfg.kwargs.items())) and cfg.num_warps == best_config.get('num_warps') and (cfg.num_stages == best_config.get('num_stages'))]
    if len(matching_configs) != 1:
        return None
    return matching_configs[0]