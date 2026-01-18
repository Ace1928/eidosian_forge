import operator
from typing import Any, Optional
def find_prefixsum_idx(self, prefixsum: float) -> int:
    """Finds highest i, for which: sum(arr[0]+..+arr[i - i]) <= prefixsum.

        Args:
            prefixsum: `prefixsum` upper bound in above constraint.

        Returns:
            int: Largest possible index (i) satisfying above constraint.
        """
    assert 0 <= prefixsum <= self.sum() + 1e-05
    idx = 1
    while idx < self.capacity:
        update_idx = 2 * idx
        if self.value[update_idx] > prefixsum:
            idx = update_idx
        else:
            prefixsum -= self.value[update_idx]
            idx = update_idx + 1
    return idx - self.capacity