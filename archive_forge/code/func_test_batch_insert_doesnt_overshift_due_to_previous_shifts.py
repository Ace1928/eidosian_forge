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
def test_batch_insert_doesnt_overshift_due_to_previous_shifts():
    a = cirq.NamedQubit('a')
    c = cirq.Circuit([cirq.H(a)] * 3)
    c.batch_insert([(0, cirq.Z(a)), (0, cirq.Z(a)), (1, cirq.X(a)), (2, cirq.Y(a))])
    assert c == cirq.Circuit(cirq.Z(a), cirq.Z(a), cirq.H(a), cirq.X(a), cirq.H(a), cirq.Y(a), cirq.H(a))