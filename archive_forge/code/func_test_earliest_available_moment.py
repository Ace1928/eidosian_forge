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
def test_earliest_available_moment():
    q = cirq.LineQubit.range(3)
    c = cirq.Circuit(cirq.Moment(cirq.measure(q[0], key='m')), cirq.Moment(cirq.X(q[1]).with_classical_controls('m')))
    assert c.earliest_available_moment(cirq.Y(q[0])) == 1
    assert c.earliest_available_moment(cirq.Y(q[1])) == 2
    assert c.earliest_available_moment(cirq.Y(q[2])) == 0
    assert c.earliest_available_moment(cirq.Y(q[2]).with_classical_controls('m')) == 1
    assert c.earliest_available_moment(cirq.Y(q[2]).with_classical_controls('m'), end_moment_index=1) == 1
    assert c.earliest_available_moment(cirq.Y(q[1]).with_classical_controls('m'), end_moment_index=1) == 1