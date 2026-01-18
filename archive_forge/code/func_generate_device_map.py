import math
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union
from .state import PartialState
from .utils import (
def generate_device_map(model, num_processes: int=1, no_split_module_classes=None, max_memory: dict=None):
    """
    Calculates the device map for `model` with an offset for PiPPy
    """
    if num_processes == 1:
        return infer_auto_device_map(model, no_split_module_classes=no_split_module_classes, clean_result=False)
    if max_memory is None:
        model_size, shared = calculate_maximum_sizes(model)
        memory = (model_size + shared[0]) / num_processes
        memory = convert_bytes(memory)
        value, ending = memory.split(' ')
        memory = math.ceil(float(value)) * 1.1
        memory = f'{memory} {ending}'
        max_memory = {i: memory for i in range(num_processes)}
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes, clean_result=False)
    return device_map