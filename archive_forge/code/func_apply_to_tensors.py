from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence
def apply_to_tensors(fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple, Set]) -> Any:
    """Recursively apply to all tensor in different kinds of container types."""
    return apply_to_type(torch.is_tensor, fn, container)