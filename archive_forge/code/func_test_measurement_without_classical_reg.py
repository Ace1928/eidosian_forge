import numpy as np
import pytest
from pyquil import Program
from pyquil.simulation.tools import program_unitary
from cirq import Circuit, LineQubit
from cirq_rigetti.quil_input import (
from cirq.ops import (
def test_measurement_without_classical_reg():
    """Measure operations must declare a classical register."""
    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil('MEASURE 0')