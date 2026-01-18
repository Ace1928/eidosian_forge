import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mutable_frozen_copy():
    a, b, c = cirq.LineQubit.range(3)
    p = -cirq.X(a) * cirq.Y(b) * cirq.Z(c)
    pf = p.frozen()
    pm = p.mutable_copy()
    pmm = pm.mutable_copy()
    pmf = pm.frozen()
    assert isinstance(p, cirq.PauliString)
    assert isinstance(pf, cirq.PauliString)
    assert isinstance(pm, cirq.MutablePauliString)
    assert isinstance(pmm, cirq.MutablePauliString)
    assert isinstance(pmf, cirq.PauliString)
    assert p is pf
    assert pm is not pmm
    assert p == pf == pm == pmm == pmf