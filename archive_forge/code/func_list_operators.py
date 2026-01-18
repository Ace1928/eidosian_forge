import functools
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple
import torch
from torch.testing._internal.composite_compliance import (
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
def list_operators(function, *args, **kwargs):
    """
    Returns the list of operators used inside `function` with
    *args and **kwargs
    """
    verbose_mode = VerboseTorchDispatchMode()
    with verbose_mode:
        function(*args, **kwargs)
    return verbose_mode.operators