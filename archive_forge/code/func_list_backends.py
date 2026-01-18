import functools
import sys
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple
import torch
from torch import fx
def list_backends(exclude_tags=('debug', 'experimental')) -> List[str]:
    """
    Return valid strings that can be passed to:

        torch.compile(..., backend="name")
    """
    _lazy_import()
    exclude_tags = set(exclude_tags or ())
    return sorted([name for name, backend in _BACKENDS.items() if not exclude_tags.intersection(backend._tags)])