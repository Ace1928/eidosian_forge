import itertools
import random
from typing import List, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('cv', [[1] * 3, random_cv(5), random_cv(6), random_cv(7)])
@allow_deprecated_cirq_ft_use_in_tests
def test_multi_controlled_and_gate(cv: List[int]):
    gate = cirq_ft.And(cv)
    r = gate.signature
    assert r.get_right('junk').total_bits() == r.get_left('ctrl').total_bits() - 2
    quregs = infra.get_named_qubits(r)
    and_op = gate.on_registers(**quregs)
    circuit = cirq.Circuit(and_op)
    input_controls = [cv] + [random_cv(len(cv)) for _ in range(10)]
    qubit_order = infra.merge_qubits(gate.signature, **quregs)
    for input_control in input_controls:
        initial_state = input_control + [0] * (r.get_right('junk').total_bits() + 1)
        result = cirq.Simulator(dtype=np.complex128).simulate(circuit, initial_state=initial_state, qubit_order=qubit_order)
        expected_output = np.asarray([0, 1] if input_control == cv else [1, 0])
        assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(result.final_state_vector, keep_indices=[cirq.num_qubits(gate) - 1]), expected_output)
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit + cirq.Circuit(cirq.inverse(and_op)), qubit_order=qubit_order, inputs=initial_state, outputs=initial_state)