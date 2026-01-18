from typing import cast
import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells import arithmetic_cells
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_helpers():
    f = arithmetic_cells._popcnt
    assert f(0) == 0
    assert f(1) == 1
    assert f(2) == 1
    assert f(3) == 2
    assert f(4) == 1
    assert f(5) == 2
    assert f(6) == 2
    assert f(7) == 3
    assert f(8) == 1
    assert f(9) == 2
    g = arithmetic_cells._invertible_else_1
    assert g(0, 0) == 1
    assert g(0, 1) == 0
    assert g(0, 2) == 1
    assert g(2, 0) == 1
    assert g(0, 15) == 1
    assert g(1, 15) == 1
    assert g(2, 15) == 2
    assert g(3, 15) == 1
    assert g(0, 16) == 1
    assert g(1, 16) == 1
    assert g(2, 16) == 1
    assert g(3, 16) == 3
    assert g(4, 16) == 1
    assert g(5, 16) == 5
    assert g(6, 16) == 1
    assert g(7, 16) == 7
    assert g(51, 16) == 51
    h = arithmetic_cells._mod_inv_else_1
    assert h(0, 0) == 1
    assert h(0, 1) == 0
    assert h(0, 2) == 1
    assert h(2, 0) == 1
    assert h(0, 15) == 1
    assert h(1, 15) == 1
    assert h(2, 15) == 8
    assert h(3, 15) == 1
    assert h(0, 16) == 1
    assert h(1, 16) == 1
    assert h(2, 16) == 1
    assert h(3, 16) == 11
    assert h(4, 16) == 1
    assert h(5, 16) == 13
    assert h(6, 16) == 1
    assert h(7, 16) == 7