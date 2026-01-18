import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def test_fredkin():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.FREDKIN(a, b, c))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[["•","Swap","Swap"]]}\n    ', escape_url=False)
    x, y, z = cirq.LineQubit.range(3, 6)
    circuit = cirq.Circuit(cirq.CSWAP(a, b, c), cirq.CSWAP(x, y, z))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[\n            ["•","Swap","Swap"],\n            [1,1,1,"•","Swap","Swap"]\n        ]}\n    ', escape_url=False)