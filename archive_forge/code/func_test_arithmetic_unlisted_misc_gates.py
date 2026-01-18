from typing import cast
import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells import arithmetic_cells
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_arithmetic_unlisted_misc_gates():
    assert_url_to_circuit_returns('{"cols":[["^=A3",1,1,"inputA2"]]}', maps={0: 0, 1: 5, 2: 10, 31: 19})
    assert_url_to_circuit_returns('{"cols":[["^=A2",1,"inputA3"]]}', maps={0: 0, 1: 9, 2: 18, 4: 4, 31: 7})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":5}],["^=A4"]]}', maps={0: 5, 15: 10})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":11}],["+cntA4"]]}', maps={0: 3, 1: 4, 15: 2})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":5}],["+cntA4"]]}', maps={0: 2, 1: 3, 15: 1})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":7}],["-cntA4"]]}', maps={0: 13, 1: 14, 15: 12})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":5}],["Flip<A4"]]}', maps={15: 15, 6: 6, 5: 5, 4: 0, 3: 1, 2: 2, 1: 3, 0: 4})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":6}],["Flip<A4"]]}', maps={15: 15, 6: 6, 5: 0, 4: 1, 3: 2, 2: 3, 1: 4, 0: 5})