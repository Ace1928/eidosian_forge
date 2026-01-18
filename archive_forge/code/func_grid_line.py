from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def grid_line(self, x1: int, y1: int, x2: int, y2: int, emphasize: bool=False, doubled: bool=False):
    """Adds a vertical or horizontal line from (x1, y1) to (x2, y2).

        Horizontal line is selected on equality in the second coordinate and
        vertical line is selected on equality in the first coordinate.

        Raises:
            ValueError: If line is neither horizontal nor vertical.
        """
    if x1 == x2:
        self.vertical_line(x1, y1, y2, emphasize, doubled)
    elif y1 == y2:
        self.horizontal_line(y1, x1, x2, emphasize, doubled)
    else:
        raise ValueError('Line is neither horizontal nor vertical')