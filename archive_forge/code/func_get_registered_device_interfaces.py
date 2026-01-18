import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
def get_registered_device_interfaces() -> Iterable[Tuple[str, Type[DeviceInterface]]]:
    return device_interfaces.items()