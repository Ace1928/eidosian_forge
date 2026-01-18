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
def test_insert_op_tree_newinline():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = cirq.Circuit()
    op_tree_list = [(-5, 0, [cirq.H(a), cirq.X(b)], [a, b]), (-15, 0, [cirq.CZ(a, b)], [a]), (15, 2, [cirq.H(b), cirq.X(a)], [b, a])]
    for given_index, actual_index, op_list, qubits in op_tree_list:
        c.insert(given_index, op_list, cirq.InsertStrategy.NEW_THEN_INLINE)
        for i in range(len(op_list)):
            assert c.operation_at(qubits[i], actual_index) == op_list[i]
    c2 = cirq.Circuit()
    c2.insert(0, [cirq.CZ(a, b), cirq.H(a), cirq.X(b), cirq.H(b), cirq.X(a)], cirq.InsertStrategy.NEW_THEN_INLINE)
    assert c == c2