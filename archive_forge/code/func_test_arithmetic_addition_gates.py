from typing import cast
import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells import arithmetic_cells
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_arithmetic_addition_gates():
    assert_url_to_circuit_returns('{"cols":[["inc3"]]}', diagram='\n0: ───Quirk(inc3)───\n      │\n1: ───#2────────────\n      │\n2: ───#3────────────\n            ', maps={0: 1, 3: 4, 7: 0})
    assert_url_to_circuit_returns('{"cols":[["dec3"]]}', maps={0: 7, 3: 2, 7: 6})
    assert_url_to_circuit_returns('{"cols":[["+=A2",1,"inputA2"]]}', maps={0: 0, 6: 14, 11: 7})
    assert_url_to_circuit_returns('{"cols":[["-=A2",1,"inputA2"]]}', maps={0: 0, 6: 14, 11: 15})