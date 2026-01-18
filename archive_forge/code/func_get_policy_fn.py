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
def get_policy_fn(self, *args, **kwargs):
    if self.policy_fn is None:
        self.policy_fn = self._get_policy_fn(*args, **kwargs)
    return self.policy_fn