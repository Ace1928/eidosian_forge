import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
def get_interface_for_device(device: str) -> Type[DeviceInterface]:
    if device in device_interfaces:
        return device_interfaces[device]
    raise NotImplementedError(f'No interface for device {device}')