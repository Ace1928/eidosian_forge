from typing import Iterator, Callable
import sympy
import cirq
from cirq.interop.quirk.cells.cell import CellMaker, CELL_SIZES, ExplicitOperationsCell
def generate_all_frequency_space_cell_makers() -> Iterator[CellMaker]:
    yield from _family('QFT', lambda n: cirq.QuantumFourierTransformGate(n))
    yield from _family('QFT†', lambda n: cirq.inverse(cirq.QuantumFourierTransformGate(n)))
    yield from _family('PhaseGradient', lambda n: cirq.PhaseGradientGate(num_qubits=n, exponent=0.5))
    yield from _family('PhaseUngradient', lambda n: cirq.PhaseGradientGate(num_qubits=n, exponent=-0.5))
    yield from _family('grad^t', lambda n: cirq.PhaseGradientGate(num_qubits=n, exponent=2 ** (n - 1) * sympy.Symbol('t')))
    yield from _family('grad^-t', lambda n: cirq.PhaseGradientGate(num_qubits=n, exponent=-2 ** (n - 1) * sympy.Symbol('t')))