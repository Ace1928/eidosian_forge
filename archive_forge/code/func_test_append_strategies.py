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
def test_append_strategies():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    stream = [cirq.X(a), cirq.CZ(a, b), cirq.X(b), cirq.X(b), cirq.X(a)]
    c = cirq.Circuit()
    c.append(stream, cirq.InsertStrategy.NEW)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(b)]), cirq.Moment([cirq.X(b)]), cirq.Moment([cirq.X(a)])])
    c = cirq.Circuit()
    c.append(stream, cirq.InsertStrategy.INLINE)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(b)]), cirq.Moment([cirq.X(b), cirq.X(a)])])
    c = cirq.Circuit()
    c.append(stream, cirq.InsertStrategy.EARLIEST)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(b), cirq.X(a)]), cirq.Moment([cirq.X(b)])])