from typing import List, Sequence, Tuple
import numpy as np
import sympy
import cirq
from cirq.contrib.custom_simulators.custom_state_simulator import CustomStateSimulator
def test_basis_state_simulator():
    sim = CustomStateSimulator(ComputationalBasisSimState)
    circuit = create_test_circuit()
    r = sim.simulate(circuit)
    assert r.measurements == {'a': np.array([1]), 'b': np.array([2])}
    assert r._final_simulator_state._state.basis == [2, 2]