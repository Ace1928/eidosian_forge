import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_gate_equal_up_to_global_phase():
    groups = [[cirq.PauliStringPhasorGate(dps_x, exponent_neg=0.25), cirq.PauliStringPhasorGate(dps_x, exponent_neg=0, exponent_pos=-0.25), cirq.PauliStringPhasorGate(dps_x, exponent_pos=-0.125, exponent_neg=0.125)], [cirq.PauliStringPhasorGate(dps_x)], [cirq.PauliStringPhasorGate(dps_y, exponent_neg=0.25)], [cirq.PauliStringPhasorGate(dps_xy, exponent_neg=0.25)]]
    for g1 in groups:
        for e1 in g1:
            assert not e1.equal_up_to_global_phase('not even close')
            for g2 in groups:
                for e2 in g2:
                    assert e1.equal_up_to_global_phase(e2) == (g1 is g2)