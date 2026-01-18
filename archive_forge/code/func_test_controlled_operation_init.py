import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
def test_controlled_operation_init():

    class G(cirq.testing.SingleQubitGate):

        def _has_mixture_(self):
            return True
    g = G()
    cb = cirq.NamedQubit('ctr')
    q = cirq.NamedQubit('q')
    v = cirq.GateOperation(g, (q,))
    c = cirq.ControlledOperation([cb], v)
    assert c.sub_operation == v
    assert c.controls == (cb,)
    assert c.qubits == (cb, q)
    assert c == c.with_qubits(cb, q)
    assert c.control_values == cirq.SumOfProducts(((1,),))
    assert cirq.qid_shape(c) == (2, 2)
    c = cirq.ControlledOperation([cb], v, control_values=[0])
    assert c.sub_operation == v
    assert c.controls == (cb,)
    assert c.qubits == (cb, q)
    assert c == c.with_qubits(cb, q)
    assert c.control_values == cirq.SumOfProducts(((0,),))
    assert cirq.qid_shape(c) == (2, 2)
    c = cirq.ControlledOperation([cb.with_dimension(3)], v)
    assert c.sub_operation == v
    assert c.controls == (cb.with_dimension(3),)
    assert c.qubits == (cb.with_dimension(3), q)
    assert c == c.with_qubits(cb.with_dimension(3), q)
    assert c.control_values == cirq.SumOfProducts(((1,),))
    assert cirq.qid_shape(c) == (3, 2)
    with pytest.raises(ValueError, match='cirq\\.num_qubits\\(control_values\\) != len\\(controls\\)'):
        _ = cirq.ControlledOperation([cb], v, control_values=[1, 1])
    with pytest.raises(ValueError, match='Control values .*outside of range'):
        _ = cirq.ControlledOperation([cb], v, control_values=[2])
    with pytest.raises(ValueError, match='Control values .*outside of range'):
        _ = cirq.ControlledOperation([cb], v, control_values=[(1, -1)])
    with pytest.raises(ValueError, match=re.escape("Duplicate control qubits ['ctr'].")):
        _ = cirq.ControlledOperation([cb, cirq.LineQubit(0), cb], cirq.X(q))
    with pytest.raises(ValueError, match=re.escape("Sub-op and controls share qubits ['ctr']")):
        _ = cirq.ControlledOperation([cb, cirq.LineQubit(0)], cirq.CX(cb, q))
    with pytest.raises(ValueError, match='Cannot control measurement'):
        _ = cirq.ControlledOperation([cb], cirq.measure(q))
    with pytest.raises(ValueError, match='Cannot control channel'):
        _ = cirq.ControlledOperation([cb], cirq.PhaseDampingChannel(1)(q))