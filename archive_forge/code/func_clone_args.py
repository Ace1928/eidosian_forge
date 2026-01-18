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
def clone_args(self, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
    from .compile_fx import clone_preserve_strides
    cloned_args = []
    for i, arg in enumerate(args):
        if self.fn.arg_names[i] in self.mutated_arg_names:
            assert isinstance(arg, torch.Tensor)
            cloned_args.append(clone_preserve_strides(arg))
        else:
            cloned_args.append(arg)
    cloned_kwargs: Dict[str, Any] = {}
    for name, arg in kwargs.items():
        if name in self.mutated_arg_names:
            assert isinstance(arg, torch.Tensor)
            cloned_kwargs[name] = clone_preserve_strides(arg)
        else:
            cloned_kwargs[name] = arg
    return (cloned_args, cloned_kwargs)