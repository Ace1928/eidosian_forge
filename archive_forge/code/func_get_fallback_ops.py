import copy
import dataclasses
import itertools
import os
from typing import Any, Callable, Dict, List
import torch
import torch._lazy as lazy
import torch._lazy.metrics as metrics
from torch import fx
from torch._lazy import computation, debug as lazy_debug
from torch._lazy.tensor_factory_functions import tensor_factory_functions
def get_fallback_ops():
    fallback_ops = []
    for opname in metrics.counter_names():
        if 'aten::' not in opname:
            continue
        val = int(metrics.counter_value(opname))
        if val > 0:
            fallback_ops.append(f'{opname}={val}')
    return fallback_ops