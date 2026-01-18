import inspect
from typing import (
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES
def generate_all_arithmetic_cell_makers() -> Iterable[CellMaker]:
    global _cached_cells
    if _cached_cells is None:
        _cached_cells = tuple(_generate_helper())
    return _cached_cells