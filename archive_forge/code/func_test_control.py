from typing import Union, Tuple, cast
import numpy as np
import pytest
import sympy
import cirq
from cirq.type_workarounds import NotImplementedType
def test_control():

    class G(cirq.testing.SingleQubitGate):

        def _has_mixture_(self):
            return True
    g = G()
    assert g.controlled() == cirq.ControlledGate(g)
    cg = g.controlled()
    assert isinstance(cg, cirq.ControlledGate)
    assert cg.sub_gate == g
    assert cg.num_controls() == 1
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(g)
    eq.add_equality_group(g.controlled(), g.controlled(control_values=[1]), g.controlled(control_qid_shape=(2,)), cirq.ControlledGate(g, num_controls=1), g.controlled(control_values=cirq.SumOfProducts([[1]])))
    eq.add_equality_group(cirq.ControlledGate(g, num_controls=2), g.controlled(control_values=[1, 1]), g.controlled(control_qid_shape=[2, 2]), g.controlled(num_controls=2), g.controlled().controlled(), g.controlled(control_values=cirq.SumOfProducts([[1, 1]])))
    eq.add_equality_group(cirq.ControlledGate(g, control_values=[0, 1]), g.controlled(control_values=[0, 1]), g.controlled(control_values=[1]).controlled(control_values=[0]), g.controlled(control_values=cirq.SumOfProducts([[1]])).controlled(control_values=[0]))
    eq.add_equality_group(g.controlled(control_values=[0]).controlled(control_values=[1]))
    eq.add_equality_group(cirq.ControlledGate(g, control_qid_shape=[4, 3]), g.controlled(control_qid_shape=[4, 3]), g.controlled(control_qid_shape=[3]).controlled(control_qid_shape=[4]))
    eq.add_equality_group(g.controlled(control_qid_shape=[4]).controlled(control_qid_shape=[3]))