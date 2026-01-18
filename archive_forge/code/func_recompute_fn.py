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
def recompute_fn(*inputs):
    kwargs, *args = inputs
    rng_devices = []
    if preserve_rng_state and had_device_in_fwd:
        rng_devices = fwd_devices
    with torch.random.fork_rng(devices=rng_devices, enabled=preserve_rng_state, device_type=device):
        if preserve_rng_state:
            torch.set_rng_state(fwd_cpu_state)
            if had_device_in_fwd:
                set_device_states(fwd_devices, fwd_device_states)
        device_autocast_ctx = device_module.amp.autocast(**device_autocast_kwargs) if _supports_autocast(device) else contextlib.nullcontext()
        with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:
            fn(*args, **kwargs)