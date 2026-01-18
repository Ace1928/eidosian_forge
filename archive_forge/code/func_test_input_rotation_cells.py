import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_input_rotation_cells():
    with pytest.raises(ValueError, match='classical constant'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["Z^(A/2^n)",{"id":"setA","arg":3}]]}')
    with pytest.raises(ValueError, match="Missing input 'a'"):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["X^(A/2^n)"]]}')
    assert_url_to_circuit_returns('{"cols":[["Z^(A/2^n)","inputA2"]]}', diagram='\n0: ───Z^(A/2^2)───\n      │\n1: ───A0──────────\n      │\n2: ───A1──────────\n        ', unitary=np.diag([1, 1, 1, 1, 1j ** 0, 1j ** 0.5, 1j ** 1, 1j ** 1.5]))
    assert_url_to_circuit_returns('{"cols":[["Z^(-A/2^n)","inputA1"]]}', unitary=np.diag([1, 1, 1, -1j]))
    assert_url_to_circuit_returns('{"cols":[["H"],["X^(A/2^n)","inputA2"],["H"]]}', unitary=np.diag([1, 1, 1, 1, 1j ** 0, 1j ** 0.5, 1j ** 1, 1j ** 1.5]))
    assert_url_to_circuit_returns('{"cols":[["H"],["X^(-A/2^n)","inputA2"],["H"]]}', unitary=np.diag([1, 1, 1, 1, 1j ** 0, 1j ** (-0.5), 1j ** (-1), 1j ** (-1.5)]))
    assert_url_to_circuit_returns('{"cols":[["X^-½"],["Y^(A/2^n)","inputA2"],["X^½"]]}', unitary=np.diag([1, 1, 1, 1, 1j ** 0, 1j ** 0.5, 1j ** 1, 1j ** 1.5]))
    assert_url_to_circuit_returns('{"cols":[["X^-½"],["Y^(-A/2^n)","inputA2"],["X^½"]]}', unitary=np.diag([1, 1, 1, 1, 1j ** 0, 1j ** (-0.5), 1j ** (-1), 1j ** (-1.5)]))
    assert_url_to_circuit_returns('{"cols":[["•","Z^(A/2^n)","inputA2"]]}', diagram='\n0: ───@^(A/2^2)───\n      │\n1: ───@───────────\n      │\n2: ───A0──────────\n      │\n3: ───A1──────────\n        ', unitary=np.diag([1 + 0j] * 13 + [1j ** 0.5, 1j, 1j ** 1.5]))
    assert_url_to_circuit_returns('{"cols":[["X^(-A/2^n)","inputA2"]]}', diagram='\n0: ───X^(-A/2^2)───\n      │\n1: ───A0───────────\n      │\n2: ───A1───────────\n        ')
    assert_url_to_circuit_returns('{"cols":[["•","X^(-A/2^n)","inputA2"]]}', diagram='\n0: ───@────────────\n      │\n1: ───X^(-A/2^2)───\n      │\n2: ───A0───────────\n      │\n3: ───A1───────────\n        ')
    assert_url_to_circuit_returns('{"cols":[["Z^(A/2^n)","inputA1","inputB1"],[1,1,"Z"],[1,1,"Z"]]}', unitary=np.diag([1, 1, 1, 1, 1, 1, 1j, 1j]))