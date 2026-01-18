from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def force_vertical_padding_after(self, index: int, padding: Union[int, float]) -> None:
    """Change the padding after the given row."""
    self.vertical_padding[index] = padding