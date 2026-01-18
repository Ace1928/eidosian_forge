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
def test_append_multiple():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.Circuit()
    c.append([cirq.X(a), cirq.X(b)], cirq.InsertStrategy.NEW)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(b)])])
    c = cirq.Circuit()
    c.append([cirq.X(a), cirq.X(b)], cirq.InsertStrategy.EARLIEST)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a), cirq.X(b)])])
    c = cirq.Circuit()
    c.append(cirq.X(a), cirq.InsertStrategy.EARLIEST)
    c.append(cirq.X(b), cirq.InsertStrategy.EARLIEST)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a), cirq.X(b)])])