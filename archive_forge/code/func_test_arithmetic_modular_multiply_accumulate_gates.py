from typing import cast
import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells import arithmetic_cells
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_arithmetic_modular_multiply_accumulate_gates():
    assert_url_to_circuit_returns('{"cols":[[{"id":"setR","arg":5},{"id":"setA","arg":3},{"id":"setB","arg":4}],["+ABmodR4"]]}', maps={0: 2, 1: 3, 2: 4, 3: 0, 4: 1, 5: 5, 15: 15})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setR","arg":27},{"id":"setA","arg":3},{"id":"setB","arg":5}],["-ABmodR6"]]}', maps={0: 27 - 15, 1: 27 - 14, 15: 0, 16: 1, 26: 26 - 15, 27: 27, 63: 63})