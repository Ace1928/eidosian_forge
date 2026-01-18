import typing
from typing import Any, Callable, List, Union, cast, Sequence
import torch
from torch import Tensor
import torch.cuda.comm
def find_tensor_idx(self):
    """
        Retrieves the index of first tensor found.
        """
    if self.atomic:
        return 0
    for i, value in enumerate(self._values):
        if torch.is_tensor(value):
            return i
    raise TypeError('No tensor found!')