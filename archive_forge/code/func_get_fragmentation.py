from typing import TYPE_CHECKING, Tuple, Optional
import pyglet
def get_fragmentation(self) -> float:
    """Get the fraction of area that's unlikely to ever be used, based on
        current allocation behaviour.

        This method is useful for debugging and profiling only.

        :rtype: float
        """
    if not self.strips:
        return 0.0
    possible_area = self.strips[-1].y2 * self.width
    return 1.0 - self.used_area / possible_area