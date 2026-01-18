import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_filters_identities():
    q1, q2 = cirq.LineQubit.range(2)
    assert cirq.PauliString({q1: cirq.I, q2: cirq.X}) == cirq.PauliString({q2: cirq.X})