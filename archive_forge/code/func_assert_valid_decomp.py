import itertools
import numpy as np
import pytest
import cirq
import sympy
def assert_valid_decomp(u_target, operations, *, single_qubit_gate_types=(cirq.ZPowGate, cirq.XPowGate, cirq.YPowGate), two_qubit_gate=cirq.SQRT_ISWAP, atol=1e-06, rtol=0, qubit_order=cirq.LineQubit.range(2)):
    for op in operations:
        if len(op.qubits) == 0 and isinstance(op.gate, cirq.GlobalPhaseGate):
            assert False, 'Global phase operation was output when it should not.'
        elif len(op.qubits) == 1 and isinstance(op.gate, single_qubit_gate_types):
            pass
        elif len(op.qubits) == 2 and op.gate == two_qubit_gate:
            pass
        else:
            assert False, f'Disallowed operation was output: {op}'
    c = cirq.Circuit(operations)
    u_decomp = c.unitary(qubit_order)
    if not cirq.allclose_up_to_global_phase(u_decomp, u_target, atol=atol, rtol=rtol):
        cirq.testing.assert_allclose_up_to_global_phase(u_decomp, u_target, atol=0.01, rtol=0.01, err_msg='Invalid decomposition.  Unitaries are completely different.')
        worst_diff = np.max(np.abs(u_decomp - u_target))
        assert False, f'Unitaries do not match closely enough (atol={atol}, rtol={rtol}, worst element diff={worst_diff}).'