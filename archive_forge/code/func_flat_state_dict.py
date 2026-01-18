from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
def flat_state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Return the flattened state_dict."""
    assert self.is_flattened
    with self._no_auto_unflatten_state_dict():
        return self.state_dict(*args, **kwargs)