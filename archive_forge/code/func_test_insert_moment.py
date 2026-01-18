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
def test_insert_moment():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = cirq.Circuit()
    moment_list = [(-10, 0, [cirq.CZ(a, b)], a, cirq.InsertStrategy.NEW_THEN_INLINE), (-20, 0, [cirq.X(a)], a, cirq.InsertStrategy.NEW), (20, 2, [cirq.X(b)], b, cirq.InsertStrategy.INLINE), (2, 2, [cirq.H(b)], b, cirq.InsertStrategy.EARLIEST), (-3, 1, [cirq.H(a)], a, cirq.InsertStrategy.EARLIEST)]
    for given_index, actual_index, operation, qubit, strat in moment_list:
        c.insert(given_index, cirq.Moment(operation), strat)
        assert c.operation_at(qubit, actual_index) == operation[0]