from typing import List, Union
import pytest
import sympy
import numpy as np
import cirq
import cirq_google
def test_fsim_gate_family_convert_rejects():
    for gate in [cirq.rx(np.pi / 2), cirq.CNOT, cirq.CCNOT]:
        assert cirq_google.FSimGateFamily().convert(gate, cirq.PhasedFSimGate) is None
        assert gate not in cirq_google.FSimGateFamily(gates_to_accept=[cirq.PhasedFSimGate])
    assert UnequalSycGate() not in cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC])
    assert UnequalSycGate(is_parameterized=True) not in cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC], allow_symbols=True)
    assert cirq.FSimGate(THETA, np.pi / 2) not in cirq_google.FSimGateFamily(gates_to_accept=[cirq.PhasedISwapPowGate(exponent=0.5, phase_exponent=0.1), cirq.CZPowGate])