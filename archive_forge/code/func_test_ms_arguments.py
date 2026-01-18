import numpy as np
import pytest
import sympy
import cirq
def test_ms_arguments():
    eq_tester = cirq.testing.EqualsTester()
    eq_tester.add_equality_group(cirq.ms(np.pi / 2), cirq.ops.MSGate(rads=np.pi / 2), cirq.XXPowGate(global_shift=-0.5))
    eq_tester.add_equality_group(cirq.ms(np.pi / 4), cirq.XXPowGate(exponent=0.5, global_shift=-0.5))
    eq_tester.add_equality_group(cirq.XX)
    eq_tester.add_equality_group(cirq.XX ** 0.5)