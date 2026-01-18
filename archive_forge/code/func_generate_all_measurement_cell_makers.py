from typing import Iterator, Optional, cast, Iterable, TYPE_CHECKING
from cirq import ops
from cirq.interop.quirk.cells.cell import CellMaker, ExplicitOperationsCell
def generate_all_measurement_cell_makers() -> Iterator[CellMaker]:
    yield _measurement('Measure')
    yield _measurement('ZDetector')
    yield _measurement('YDetector', basis_change=ops.X ** (-0.5))
    yield _measurement('XDetector', basis_change=ops.Y ** 0.5)