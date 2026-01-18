from typing import cast
import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells import arithmetic_cells
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_arithmetic_multiply_accumulate_gates():
    assert_url_to_circuit_returns('{"cols":[["+=AA4",1,1,1,"inputA2"]]}', maps={0: 0, 18: 34, 35: 7})
    assert_url_to_circuit_returns('{"cols":[["-=AA4",1,1,1,"inputA2"]]}', maps={0: 0, 18: 2, 35: 63})
    assert_url_to_circuit_returns('{"cols":[["+=AB3",1,1,"inputA2",1,"inputB2"]]}', maps={0: 0, 14: 110, 79: 95})
    assert_url_to_circuit_returns('{"cols":[["-=AB3",1,1,"inputA2",1,"inputB2"]]}', maps={0: 0, 14: 46, 79: 63})