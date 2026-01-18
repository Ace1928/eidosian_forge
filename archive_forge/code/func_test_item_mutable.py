import numbers
from typing import List
import numpy as np
import pytest
import sympy
import cirq
from cirq.ops.dense_pauli_string import _vectorized_pauli_mul_phase
def test_item_mutable():
    m = cirq.MutableDensePauliString
    p = m('XYIZ', coefficient=-1)
    assert p[-1] == cirq.Z
    assert p[0] == cirq.X
    assert p[1] == cirq.Y
    assert p[2] == cirq.I
    assert p[3] == cirq.Z
    with pytest.raises(IndexError):
        _ = p[4]
    with pytest.raises(TypeError):
        _ = p['test']
    with pytest.raises(TypeError):
        p['test'] = 'X'
    p[2] = cirq.X
    assert p == m('XYXZ', coefficient=-1)
    p[3] = 'X'
    p[0] = 'I'
    assert p == m('IYXX', coefficient=-1)
    p[2:] = p[:2]
    assert p == m('IYIY', coefficient=-1)
    p[2:] = 'ZZ'
    assert p == m('IYZZ', coefficient=-1)
    p[2:] = 'IY'
    assert p == m('IYIY', coefficient=-1)
    q = p[:2]
    assert q == m('IY')
    q[0] = cirq.Z
    assert q == m('ZY')
    assert p == m('ZYIY', coefficient=-1)
    with pytest.raises(ValueError, match='coefficient is not 1'):
        p[:] = p
    assert p[:] == m('ZYIY')
    assert p[1:] == m('YIY')
    assert p[::2] == m('ZI')
    p[2:] = 'XX'
    assert p == m('ZYXX', coefficient=-1)