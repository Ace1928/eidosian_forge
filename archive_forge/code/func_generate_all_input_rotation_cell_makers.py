from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import cirq
from cirq import ops, linalg, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
def generate_all_input_rotation_cell_makers() -> Iterator[CellMaker]:
    yield _input_rotation_gate('X^(A/2^n)', ops.X, +1)
    yield _input_rotation_gate('Y^(A/2^n)', ops.Y, +1)
    yield _input_rotation_gate('Z^(A/2^n)', ops.Z, +1)
    yield _input_rotation_gate('X^(-A/2^n)', ops.X, -1)
    yield _input_rotation_gate('Y^(-A/2^n)', ops.Y, -1)
    yield _input_rotation_gate('Z^(-A/2^n)', ops.Z, -1)