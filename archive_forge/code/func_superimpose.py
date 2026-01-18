from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def superimpose(self, other: 'cirq.TextDiagramDrawer') -> 'cirq.TextDiagramDrawer':
    self.entries.update(other.entries)
    self.horizontal_lines += other.horizontal_lines
    self.vertical_lines += other.vertical_lines
    self.horizontal_padding.update(other.horizontal_padding)
    self.vertical_padding.update(other.vertical_padding)
    return self