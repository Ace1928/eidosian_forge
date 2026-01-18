import typing
from typing import Callable, Iterable, Iterator, List, Tuple, Union, cast
import torch
from torch import Tensor
import torch.cuda.comm
@property
def tensor_or_tensors(self) -> TensorOrTensors:
    """Retrieves the underlying tensor or tensors regardless of type."""
    return self.value