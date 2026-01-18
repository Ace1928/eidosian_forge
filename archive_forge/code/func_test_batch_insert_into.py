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
def test_batch_insert_into():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    original = cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    after.batch_insert_into([])
    assert after == original
    after = original.copy()
    after.batch_insert_into([(0, cirq.X(b))])
    assert after == cirq.Circuit([cirq.Moment([cirq.X(a), cirq.X(b)]), cirq.Moment(), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    after.batch_insert_into([(0, [cirq.X(b), cirq.X(c)])])
    assert after == cirq.Circuit([cirq.Moment([cirq.X(a), cirq.X(b), cirq.X(c)]), cirq.Moment(), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    after.batch_insert_into([(1, cirq.Z(b))])
    assert after == cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Z(b)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    after.batch_insert_into([(1, [cirq.Z(a), cirq.Z(b)])])
    assert after == cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Z(a), cirq.Z(b)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    after.batch_insert_into([(1, cirq.Z(b)), (0, cirq.X(b))])
    assert after == cirq.Circuit([cirq.Moment([cirq.X(a), cirq.X(b)]), cirq.Moment([cirq.Z(b)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    with pytest.raises(IndexError):
        after.batch_insert_into([(500, cirq.X(a))])
    assert after == original
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_insert_into([(0, cirq.X(a))])
    assert after == original
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_insert_into([(0, [cirq.X(b), cirq.X(c), cirq.X(a)])])
    assert after == original
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_insert_into([(1, cirq.X(a)), (1, cirq.CZ(a, b))])
    assert after == original