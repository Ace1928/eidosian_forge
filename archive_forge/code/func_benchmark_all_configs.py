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
@dynamo_timed
def benchmark_all_configs(self, *args, **kwargs):
    timings = {launcher: self.bench(launcher, *args, **kwargs) for launcher in self.launchers}
    for k, v in timings.items():
        self.coordesc_tuner.cache_benchmark_result(k.config, v)
    if log.isEnabledFor(logging.DEBUG):
        log.debug('Benchmark all input configs get:')
        for k, v in timings.items():
            log.debug('%s: %f, nreg %d, nspill %d, #shared-mem %d', k.config, v, k.n_regs, k.n_spills, k.shared)
    return timings