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
def test_batch_insert_maintains_order_despite_multiple_previous_inserts():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(a))
    c.batch_insert([(0, cirq.Z(a)), (0, cirq.Z(a)), (0, cirq.Z(a)), (1, cirq.CZ(a, b))])
    assert c == cirq.Circuit([cirq.Z(a)] * 3, cirq.H(a), cirq.CZ(a, b))