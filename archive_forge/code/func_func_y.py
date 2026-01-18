from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def func_y(y: Union[int, float]) -> Union[int, float]:
    return func(0, y)[1]