from typing import List, Tuple
import re
import numpy as np
import pytest
import sympy
import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound
def test_diagram_period():

    class ShiftyGate(cirq.EigenGate, cirq.testing.SingleQubitGate):

        def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
            raise NotImplementedError()

        def __init__(self, e, *shifts):
            super().__init__(exponent=e, global_shift=np.random.random())
            self.shifts = shifts

        def _eigen_shifts(self):
            return list(self.shifts)
    args = cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT
    assert ShiftyGate(0.5, 0, 1)._diagram_exponent(args) == 0.5
    assert ShiftyGate(1.5, 0, 1)._diagram_exponent(args) == -0.5
    assert ShiftyGate(2.5, 0, 1)._diagram_exponent(args) == 0.5
    assert ShiftyGate(0.5, 0.5, -0.5)._diagram_exponent(args) == 0.5
    assert ShiftyGate(1.5, 0.5, -0.5)._diagram_exponent(args) == -0.5
    assert ShiftyGate(2.5, 0.5, -0.5)._diagram_exponent(args) == 0.5
    np.testing.assert_allclose(ShiftyGate(np.e, 0, 1 / np.e)._diagram_exponent(args), np.e, atol=0.01)
    np.testing.assert_allclose(ShiftyGate(np.e * 2.5, 0, 1 / np.e)._diagram_exponent(args), np.e / 2, atol=0.01)
    assert ShiftyGate(505.2, 0, np.pi, np.e)._diagram_exponent(args) == 505.2