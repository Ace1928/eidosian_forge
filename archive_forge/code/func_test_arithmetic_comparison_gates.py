from typing import cast
import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells import arithmetic_cells
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_arithmetic_comparison_gates():
    with pytest.raises(ValueError, match='Missing input'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["^A<B"]]}')
    assert_url_to_circuit_returns('{"cols":[["^A<B","inputA2",1,"inputB2"]]}', diagram='\n0: ───Quirk(^A<B)───\n      │\n1: ───A0────────────\n      │\n2: ───A1────────────\n      │\n3: ───B0────────────\n      │\n4: ───B1────────────\n        ', maps={2: 18, 18: 2, 14: 14, 10: 10, 6: 22})
    assert_url_to_circuit_returns('{"cols":[["^A>B","inputA2",1,"inputB2"]]}', maps={14: 30, 10: 10, 6: 6})
    assert_url_to_circuit_returns('{"cols":[["^A>=B","inputA2",1,"inputB2"]]}', maps={14: 30, 10: 26, 6: 6})
    assert_url_to_circuit_returns('{"cols":[["^A<=B","inputA2",1,"inputB2"]]}', maps={14: 14, 10: 26, 6: 22})
    assert_url_to_circuit_returns('{"cols":[["^A=B","inputA2",1,"inputB2"]]}', maps={14: 14, 10: 26, 6: 6})
    assert_url_to_circuit_returns('{"cols":[["^A!=B","inputA2",1,"inputB2"]]}', maps={14: 30, 10: 10, 6: 22})