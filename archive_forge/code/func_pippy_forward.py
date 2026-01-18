import math
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union
from .state import PartialState
from .utils import (
def pippy_forward(forward, num_chunks, gather_output, *args, **kwargs):
    state = PartialState()
    output = None
    if state.num_processes == 1:
        output = forward(*args, **kwargs)
    elif state.is_local_main_process:
        found_batch_size = find_pippy_batch_size(args, kwargs)
        if found_batch_size is None:
            raise ValueError('Could not find batch size from args or kwargs')
        elif found_batch_size != num_chunks:
            args = pad_input_tensors(args, found_batch_size, num_chunks)
            kwargs = pad_input_tensors(kwargs, found_batch_size, num_chunks)
        forward(*args, **kwargs)
    elif state.is_last_process:
        output = forward()
    else:
        forward()
    if gather_output:
        output = copy_tensor_to_devices(output)
    return output