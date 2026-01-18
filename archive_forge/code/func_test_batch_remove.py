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
def test_batch_remove():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Z(b)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    after.batch_remove([])
    assert after == original
    after = original.copy()
    after.batch_remove([(0, cirq.X(a))])
    assert after == cirq.Circuit([cirq.Moment(), cirq.Moment([cirq.Z(b)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    with pytest.raises(IndexError):
        after.batch_remove([(500, cirq.X(a))])
    assert after == original
    after = original.copy()
    after.batch_remove([(0, cirq.X(a)), (2, cirq.CZ(a, b))])
    assert after == cirq.Circuit([cirq.Moment(), cirq.Moment([cirq.Z(b)]), cirq.Moment(), cirq.Moment([cirq.X(a), cirq.X(b)])])
    after = original.copy()
    after.batch_remove([(0, cirq.X(a)), (1, cirq.Z(b)), (2, cirq.CZ(a, b)), (3, cirq.X(a)), (3, cirq.X(b))])
    assert after == cirq.Circuit([cirq.Moment(), cirq.Moment(), cirq.Moment(), cirq.Moment()])
    after = original.copy()
    after.batch_remove([(3, cirq.X(a))])
    assert after == cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Z(b)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(b)])])
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_remove([(0, cirq.X(b))])
    assert after == original
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_remove([(0, cirq.X(a)), (0, cirq.X(a))])
    assert after == original