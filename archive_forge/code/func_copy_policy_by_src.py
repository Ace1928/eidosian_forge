from typing import Dict, Iterable, List, Tuple
from torch import nn
from .namespace import Namespace
def copy_policy_by_src(self, prev_j: int) -> Iterable[Tuple[int, Namespace, str]]:
    """Generates skip routes for the given destination partition number.
        The skip routes are sorted by source partition number in ascending
        order.

        Yields:
            Each tuple of (source partition number, namespace, name).

        """
    for next_j, ns, name in self.by_src_partition[prev_j]:
        if prev_j == next_j:
            continue
        yield (next_j, ns, name)