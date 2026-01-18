from __future__ import annotations
import abc
from monty.json import MSONable
def everything_equal(self, other):
    """Compare with another environment node.

        Returns:
            bool: True if it is equal to the other node.
        """
    return super().everything_equal(other) and self.coordination_environment == other.coordination_environment