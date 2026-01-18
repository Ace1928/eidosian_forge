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
def clone_inputs(preserve_requires_grad: bool):
    inputs: List[Union[torch.Tensor, List[torch.Tensor]]] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            inputs.append(clone_tensor(arg, preserve_requires_grad))
        elif is_iterable_of_tensors(arg):
            inputs.append([clone_tensor(t, preserve_requires_grad) for t in arg])
        else:
            inputs.append(arg)
    return inputs