import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule, map_arg
from torch.fx.graph import (
from .custom_config import PrepareCustomConfig
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from dataclasses import dataclass
from collections import namedtuple
import operator
import warnings
def assert_and_get_unique_device(module: torch.nn.Module) -> Any:
    """
    Returns the unique device for a module, or None if no device is found.
    Throws an error if multiple devices are detected.
    """
    devices = {p.device for p in module.parameters()} | {p.device for p in module.buffers()}
    '\n    As a temp workaround for AIMP HHC publish we added CPU check.remove it later. T163614564\n    '
    if {torch.device('cpu'), torch.device('meta')} == devices:
        warnings.warn("Both 'meta' and 'cpu' are present in the list of devices. Module can have one device. We Select 'cpu'.")
        devices = {torch.device('cpu')}
    ''
    assert len(devices) <= 1, f'prepare only works with cpu or single-device CUDA modules, but got devices {devices}'
    device = next(iter(devices)) if len(devices) > 0 else None
    return device