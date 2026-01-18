import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_conjugated_by_move_into_uninvolved():
    a, b, c, d = cirq.LineQubit.range(4)
    p = cirq.X(a) * cirq.Z(b)
    assert p.conjugated_by([cirq.SWAP(c, d), cirq.SWAP(b, c)]) == cirq.X(a) * cirq.Z(d)
    assert p.conjugated_by([cirq.SWAP(b, c), cirq.SWAP(c, d)]) == cirq.X(a) * cirq.Z(c)