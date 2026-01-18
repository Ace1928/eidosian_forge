from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def shifted(self, dx: int=0, dy: int=0) -> 'cirq.TextDiagramDrawer':
    return self.copy().shift(dx, dy)