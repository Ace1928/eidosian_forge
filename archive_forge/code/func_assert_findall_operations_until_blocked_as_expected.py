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
def assert_findall_operations_until_blocked_as_expected(circuit=None, start_frontier=None, is_blocker=None, expected_ops=None):
    if circuit is None:
        circuit = cirq.Circuit()
    if start_frontier is None:
        start_frontier = {}
    kwargs = {} if is_blocker is None else {'is_blocker': is_blocker}
    found_ops = circuit.findall_operations_until_blocked(start_frontier, **kwargs)
    for i, op in found_ops:
        assert i >= min((start_frontier[q] for q in op.qubits if q in start_frontier), default=0)
        assert set(op.qubits).intersection(start_frontier)
    if expected_ops is None:
        return
    assert sorted(found_ops) == sorted(expected_ops)