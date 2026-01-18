import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('x', [*range(4)])
@pytest.mark.parametrize('y', [*range(4)])
@allow_deprecated_cirq_ft_use_in_tests
def test_bi_qubits_mixer(x: int, y: int):
    g = cirq_ft.algos.BiQubitsMixer()
    qubits = cirq.LineQid.range(7, dimension=2)
    c = cirq.Circuit(g.on(*qubits))
    x_1, x_0 = (x >> 1 & 1, x & 1)
    y_1, y_0 = (y >> 1 & 1, y & 1)
    initial_state = [x_1, x_0, y_1, y_0, 0, 0, 0]
    result = cirq.Simulator().simulate(c, initial_state=initial_state, qubit_order=qubits).dirac_notation()[1:-1]
    x_0, y_0 = (int(result[1]), int(result[3]))
    assert np.sign(x - y) == np.sign(x_0 - y_0)
    c = cirq.Circuit(g.on(*qubits), (g ** (-1)).on(*qubits))
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(c, sorted(c.all_qubits()), initial_state, initial_state)