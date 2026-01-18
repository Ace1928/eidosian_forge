import abc
import contextlib
import weakref
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
@property
@abc.abstractmethod
def next_functions(self) -> Tuple[Tuple[Optional['Node'], int], ...]:
    ...