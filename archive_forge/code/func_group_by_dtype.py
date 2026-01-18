import collections
import logging
import sys
from typing import Any, Dict, List, MutableMapping, Set, Tuple
import torch
import torch.distributed as dist
def group_by_dtype(tensors: List[torch.Tensor]) -> Dict[torch.dtype, List[torch.Tensor]]:
    """
    Returns a dict mapping from the tensor dtype to a list containing all
    tensors of that dtype.
    Arg:
        tensors (Iterable[Tensor]): list of tensors
    """
    tensors_by_dtype = collections.defaultdict(list)
    for tensor in tensors:
        tensors_by_dtype[tensor.dtype].append(tensor)
    return tensors_by_dtype