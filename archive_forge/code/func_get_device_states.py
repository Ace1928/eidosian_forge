import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from itertools import count
from typing import (
from weakref import ReferenceType
import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    fwd_device_ids = list({arg.get_device() for arg in args if isinstance(arg, torch.Tensor) and (not arg.device.type == 'cpu')})
    fwd_device_states = []
    device_module = _get_device_module(_infer_device_type(*args))
    for device_id in fwd_device_ids:
        with device_module.device(device_id):
            fwd_device_states.append(device_module.get_rng_state())
    return (fwd_device_ids, fwd_device_states)