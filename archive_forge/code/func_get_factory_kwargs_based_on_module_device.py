from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def get_factory_kwargs_based_on_module_device():
    assert isinstance(module, torch.nn.Module)
    devices = {p.device for p in module.parameters()} | {p.device for p in module.buffers()}
    device = next(iter(devices)) if len(devices) > 0 else None
    return None if device is None else {'device': device}