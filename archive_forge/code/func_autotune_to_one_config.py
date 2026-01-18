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
def autotune_to_one_config(self, *args, **kwargs):
    """Do the actual autotuning"""
    timings = self.benchmark_all_configs(*args, **kwargs)
    self.launchers = [builtins.min(timings, key=timings.get)]
    if self.save_cache_hook:
        self.save_cache_hook(self.launchers[0].config)