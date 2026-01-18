import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_grouping_does_not_overlap():
    q0, q1 = cirq.LineQubit.range(2)
    mps_simulator = ccq.mps_simulator.MPSSimulator(grouping={q0: 0})
    with pytest.raises(ValueError, match='Grouping must cover exactly the qubits'):
        mps_simulator.simulate(cirq.Circuit(), qubit_order={q0: 0, q1: 1})