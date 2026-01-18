from typing import Any, cast, Optional, Type
import numpy as np
from cirq.circuits.circuit import Circuit
from cirq.devices import LineQubit
from cirq.ops import common_gates
from cirq.ops.dense_pauli_string import DensePauliString
from cirq import protocols
from cirq.qis import clifford_tableau
from cirq.sim import state_vector_simulation_state, final_state_vector
from cirq.sim.clifford import (
def assert_all_implemented_act_on_effects_match_unitary(val: Any, assert_tableau_implemented: bool=False, assert_ch_form_implemented: bool=False) -> None:
    """Uses val's effect on final_state_vector to check act_on(val)'s behavior.

    Checks that act_on with CliffordTableau or StabilizerStateCHForm behaves
    consistently with act_on through final state vector. Does not work with
    Operations or Gates expecting non-qubit Qids. If either of the
    assert_*_implemented args is true, fails if the corresponding method is not
    implemented for the test circuit.

    Args:
        val: A gate or operation that may be an input to protocols.act_on.
        assert_tableau_implemented: asserts that protocols.act_on() works with
          val and CliffordTableauSimulationState inputs.
        assert_ch_form_implemented: asserts that protocols.act_on() works with
          val and StabilizerChFormSimulationState inputs.
    """
    __tracebackhide__ = True
    num_qubits_val = protocols.num_qubits(val)
    if protocols.is_parameterized(val) or not protocols.has_unitary(val) or protocols.qid_shape(val) != (2,) * num_qubits_val:
        if assert_tableau_implemented or assert_ch_form_implemented:
            assert False, f'Could not assert if any act_on methods were implemented. Operating on qudits or with a non-unitary or parameterized operation is unsupported.\n\nval: {val!r}'
        return None
    qubits = LineQubit.range(num_qubits_val * 2)
    qubit_map = {qubit: i for i, qubit in enumerate(qubits)}
    circuit = Circuit()
    for i in range(num_qubits_val):
        circuit.append([common_gates.H(qubits[i]), common_gates.CNOT(qubits[i], qubits[-i - 1])])
    if hasattr(val, 'on'):
        circuit.append(val.on(*qubits[:num_qubits_val]))
    else:
        circuit.append(val.with_qubits(*qubits[:num_qubits_val]))
    state_vector = np.reshape(final_state_vector(circuit, qubit_order=qubits), protocols.qid_shape(qubits))
    tableau = _final_clifford_tableau(circuit, qubit_map)
    if tableau is None:
        assert not assert_tableau_implemented, f'Failed to generate final tableau for the test circuit.\n\nval: {val!r}'
    else:
        assert all((state_vector_has_stabilizer(state_vector, stab) for stab in tableau.stabilizers())), f'act_on clifford tableau is not consistent with final_state_vector simulation.\n\nval: {val!r}'
    stabilizer_ch_form = _final_stabilizer_state_ch_form(circuit, qubit_map)
    if stabilizer_ch_form is None:
        assert not assert_ch_form_implemented, f'Failed to generate final stabilizer state CH form for the test circuit.\n\nval: {val!r}'
    else:
        np.testing.assert_allclose(np.reshape(stabilizer_ch_form.state_vector(), protocols.qid_shape(qubits)), state_vector, atol=1e-07, err_msg=f'stabilizer_ch_form.state_vector disagrees with state_vector for {val!r}', verbose=True)