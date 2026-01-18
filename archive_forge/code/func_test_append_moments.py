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
def test_append_moments():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.Circuit()
    c.append(cirq.Moment([cirq.X(a), cirq.X(b)]), cirq.InsertStrategy.NEW)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a), cirq.X(b)])])
    c = cirq.Circuit()
    c.append([cirq.Moment([cirq.X(a), cirq.X(b)]), cirq.Moment([cirq.X(a), cirq.X(b)])], cirq.InsertStrategy.NEW)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a), cirq.X(b)]), cirq.Moment([cirq.X(a), cirq.X(b)])])