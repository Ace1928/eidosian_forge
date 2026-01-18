from collections.abc import Iterable
import warnings
from typing import Sequence
def erase_wire(self, layer: int, wire: int, length: int) -> None:
    """Erases a portion of a wire by adding a rectangle that matches the background.

        Args:
            layer (int): starting x coordinate for erasing the wire
            wire (int): y location to erase the wire from
            length (float, int): horizontal distance from ``layer`` to erase the background.

        """
    rect = patches.Rectangle((layer, wire - 0.1), length, 0.2, facecolor=plt.rcParams['figure.facecolor'], edgecolor=plt.rcParams['figure.facecolor'], zorder=1.1)
    self.ax.add_patch(rect)