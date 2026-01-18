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
def test_batch_insert_multiple_same_index():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit()
    c.batch_insert([(0, cirq.Z(a)), (0, cirq.Z(b)), (0, cirq.Z(a))])
    cirq.testing.assert_same_circuits(c, cirq.Circuit([cirq.Moment([cirq.Z(a), cirq.Z(b)]), cirq.Moment([cirq.Z(a)])]))