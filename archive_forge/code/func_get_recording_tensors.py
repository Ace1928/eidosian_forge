import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import TestCase, \
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from itertools import chain
from typing import List, Union
from torch._C import TensorType
import io
def get_recording_tensors(args):
    recording_tensors: List[torch.Tensor] = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            recording_tensors.append(arg)
        elif is_iterable_of_tensors(arg):
            recording_tensors.extend(filter(lambda t: t.requires_grad, arg))
    return recording_tensors