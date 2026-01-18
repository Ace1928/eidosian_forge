import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_indexing_by_pair(circuit_cls):
    q = cirq.LineQubit.range(3)
    c = circuit_cls([cirq.H(q[0]), cirq.H(q[1]).controlled_by(q[0]), cirq.H(q[2]).controlled_by(q[1]), cirq.X(q[0]), cirq.CCNOT(*q)])
    assert c[0, q[0]] == c[0][q[0]] == cirq.H(q[0])
    assert c[1, q[0]] == c[1, q[1]] == cirq.H(q[1]).controlled_by(q[0])
    assert c[2, q[0]] == c[2][q[0]] == cirq.X(q[0])
    assert c[2, q[1]] == c[2, q[2]] == cirq.H(q[2]).controlled_by(q[1])
    assert c[3, q[0]] == c[3, q[1]] == c[3, q[2]] == cirq.CCNOT(*q)
    with pytest.raises(KeyError, match="Moment doesn't act on given qubit"):
        _ = c[0, q[1]]
    assert c[0, q] == c[0]
    assert c[1, q] == c[1]
    assert c[2, q] == c[2]
    assert c[3, q] == c[3]
    assert c[0, q[0:2]] == c[0]
    assert c[0, q[1:3]] == cirq.Moment([])
    assert c[1, q[1:2]] == c[1]
    assert c[2, [q[0]]] == cirq.Moment([cirq.X(q[0])])
    assert c[2, q[1:3]] == cirq.Moment([cirq.H(q[2]).controlled_by(q[1])])
    assert c[np.int64(2), q[0:2]] == c[2]
    assert c[:, q[0]] == circuit_cls([cirq.Moment([cirq.H(q[0])]), cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]), cirq.Moment([cirq.X(q[0])]), cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])])])
    assert c[:, q[1]] == circuit_cls([cirq.Moment([]), cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]), cirq.Moment([cirq.H(q[2]).controlled_by(q[1])]), cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])])])
    assert c[:, q[2]] == circuit_cls([cirq.Moment([]), cirq.Moment([]), cirq.Moment([cirq.H(q[2]).controlled_by(q[1])]), cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])])])
    assert c[:, q] == c[:, q[0:2]] == c[:, [q[0], q[2]]] == c
    assert c[:, q[1:3]] == circuit_cls([cirq.Moment([]), cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]), cirq.Moment([cirq.H(q[2]).controlled_by(q[1])]), cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])])])
    assert c[1:3, q[0]] == circuit_cls([cirq.H(q[1]).controlled_by(q[0]), cirq.X(q[0])])
    assert c[1::2, q[2]] == circuit_cls([cirq.Moment([]), cirq.Moment([cirq.CCNOT(*q)])])
    assert c[0:2, q[1:3]] == circuit_cls([cirq.Moment([]), cirq.Moment([cirq.H(q[1]).controlled_by(q[0])])])
    assert c[::2, q[0:2]] == circuit_cls([cirq.Moment([cirq.H(q[0])]), cirq.Moment([cirq.H(q[2]).controlled_by(q[1]), cirq.X(q[0])])])
    assert c[0:2, q[1:3]] == c[0:2][:, q[1:3]] == c[:, q[1:3]][0:2]
    with pytest.raises(ValueError, match='If key is tuple, it must be a pair.'):
        _ = c[0, q[1], 0]
    with pytest.raises(TypeError, match='indices must be integers or slices'):
        _ = c[q[1], 0]