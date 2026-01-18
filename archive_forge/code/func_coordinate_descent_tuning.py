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
def coordinate_descent_tuning(self, launcher, *args, **kwargs):
    """
        Coordinate descent tuning can be run with or without max-autotune.

        The only difference between these two is the starting config for coordinate_descent tuning.
        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4
        and max-autotune figure out C3 is the best.

        Then if coordinate descnt tuning is run with max-autotune disabled, it will start from C1;
        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.
        """
    if self.heuristic_type == HeuristicType.TEMPLATE or self.heuristic_type == HeuristicType.USER_AUTOTUNE:
        return launcher
    cloned_args, _ = self.clone_args(*args)
    config2launcher = {launcher.config: launcher}

    def benchmark_one_config(config):
        with self.lock:
            _, launcher = self._precompile_config(config, None)
        config2launcher[config] = launcher
        out = self.bench(launcher, *cloned_args, **kwargs)
        log.debug('COORDESC: %s: %f, nreg %d, nspill %d, #shared-mem %d', launcher.config, out, launcher.n_regs, launcher.n_spills, launcher.shared)
        return out
    assert not (self.heuristic_type == HeuristicType.PERSISTENT_REDUCTION and 'RBLOCK' in launcher.config.kwargs), "Coordinate descent tuner relies on the assumption that persistent reduction's triton config does not have RBLOCK"
    best_config = self.coordesc_tuner.autotune(benchmark_one_config, launcher.config, None)
    best_config.found_by_coordesc = True
    if self.save_cache_hook:
        self.save_cache_hook(best_config, found_by_coordesc=True)
    return config2launcher.get(best_config)