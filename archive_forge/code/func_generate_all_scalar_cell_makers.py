from typing import Iterator, TYPE_CHECKING
from cirq import ops
from cirq.interop.quirk.cells.cell import CellMaker
def generate_all_scalar_cell_makers() -> Iterator[CellMaker]:
    yield _scalar('NeGate', ops.global_phase_operation(-1))
    yield _scalar('i', ops.global_phase_operation(1j))
    yield _scalar('-i', ops.global_phase_operation(-1j))
    yield _scalar('√i', ops.global_phase_operation(1j ** 0.5))
    yield _scalar('√-i', ops.global_phase_operation((-1j) ** 0.5))