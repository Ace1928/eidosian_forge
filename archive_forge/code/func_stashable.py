from typing import (
from torch import Tensor, nn
from ..microbatch import Batch
from .namespace import Namespace
from .tracker import current_skip_tracker
def stashable(self) -> Iterable[Tuple[Namespace, str]]:
    """Iterate over namespaced skip names to be stashed."""
    for name in self.stashable_names:
        yield self.namespaced(name)