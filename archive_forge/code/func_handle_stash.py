from typing import (
from torch import Tensor, nn
from ..microbatch import Batch
from .namespace import Namespace
from .tracker import current_skip_tracker
def handle_stash(name: str, tensor: Optional[Tensor]) -> None:
    if name not in self.stashable_names:
        raise RuntimeError(f"'{name}' has not been declared as stashable")
    stashed_tensors[name] = tensor