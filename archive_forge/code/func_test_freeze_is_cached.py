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
def test_freeze_is_cached():
    q = cirq.q(0)
    c = cirq.Circuit(cirq.X(q), cirq.measure(q))
    f0 = c.freeze()
    f1 = c.freeze()
    assert f1 is f0
    c.append(cirq.Y(q))
    f2 = c.freeze()
    f3 = c.freeze()
    assert f2 is not f1
    assert f3 is f2
    c[-1] = cirq.Moment(cirq.Y(q))
    f4 = c.freeze()
    f5 = c.freeze()
    assert f4 is not f3
    assert f5 is f4