import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_equal_up_to_coefficient():
    q0, = _make_qubits(1)
    assert cirq.PauliString({}, +1).equal_up_to_coefficient(cirq.PauliString({}, +1))
    assert cirq.PauliString({}, -1).equal_up_to_coefficient(cirq.PauliString({}, -1))
    assert cirq.PauliString({}, +1).equal_up_to_coefficient(cirq.PauliString({}, -1))
    assert cirq.PauliString({}, +1).equal_up_to_coefficient(cirq.PauliString({}, 2j))
    assert cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(cirq.PauliString({q0: cirq.X}, +1))
    assert cirq.PauliString({q0: cirq.X}, -1).equal_up_to_coefficient(cirq.PauliString({q0: cirq.X}, -1))
    assert cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(cirq.PauliString({q0: cirq.X}, -1))
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(cirq.PauliString({q0: cirq.Y}, +1))
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(cirq.PauliString({q0: cirq.Y}, 1j))
    assert not cirq.PauliString({q0: cirq.X}, -1).equal_up_to_coefficient(cirq.PauliString({q0: cirq.Y}, -1))
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(cirq.PauliString({q0: cirq.Y}, -1))
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(cirq.PauliString({}, +1))
    assert not cirq.PauliString({q0: cirq.X}, -1).equal_up_to_coefficient(cirq.PauliString({}, -1))
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(cirq.PauliString({}, -1))