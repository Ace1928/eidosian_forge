from typing import cast
import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells import arithmetic_cells
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_arithmetic_modular_addition_gates():
    assert_url_to_circuit_returns('{"cols":[[{"id":"setR","arg":16}],["incmodR4"]]}', diagram='\n0: ───Quirk(incmodR4,r=16)───\n      │\n1: ───#2─────────────────────\n      │\n2: ───#3─────────────────────\n      │\n3: ───#4─────────────────────\n        ', maps={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 15: 0})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setR","arg":5}],["incmodR4"]]}', maps={0: 1, 1: 2, 2: 3, 3: 4, 4: 0, 5: 5, 15: 15})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setR","arg":5}],["decmodR4"]]}', maps={0: 4, 1: 0, 2: 1, 3: 2, 4: 3, 5: 5, 15: 15})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setR","arg":5},{"id":"setA","arg":3}],["+AmodR4"]]}', maps={0: 3, 1: 4, 2: 0, 3: 1, 4: 2, 5: 5, 15: 15})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setR","arg":5},{"id":"setA","arg":3}],["-AmodR4"]]}', maps={0: 2, 1: 3, 2: 4, 3: 0, 4: 1, 5: 5, 15: 15})