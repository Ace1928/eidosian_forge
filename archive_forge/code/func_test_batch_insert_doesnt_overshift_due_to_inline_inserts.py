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
def test_batch_insert_doesnt_overshift_due_to_inline_inserts():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SWAP(a, b), cirq.SWAP(a, b), cirq.H(a), cirq.SWAP(a, b), cirq.SWAP(a, b))
    c.batch_insert([(0, cirq.X(a)), (3, cirq.X(b)), (4, cirq.Y(a))])
    assert c == cirq.Circuit(cirq.X(a), cirq.SWAP(a, b), cirq.SWAP(a, b), cirq.H(a), cirq.X(b), cirq.SWAP(a, b), cirq.Y(a), cirq.SWAP(a, b))