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
def set_device_states(devices, states) -> None:
    device_module = _get_device_module(_infer_device_type(*states))
    for device, state in zip(devices, states):
        with device_module.device(device):
            device_module.set_rng_state(state)