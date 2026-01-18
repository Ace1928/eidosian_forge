import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def test_formulaic_rotation_xyz_export():
    a = cirq.LineQubit(0)
    t = sympy.Symbol('t')
    assert_links_to(cirq.Circuit(cirq.rx(sympy.pi / 2).on(a), cirq.ry(sympy.pi * t).on(a), cirq.rz(-sympy.pi * t).on(a)), '\n        http://algassert.com/quirk#circuit={"cols":[\n            [{"arg":"(1/2)pi","id":"Rxft"}],\n            [{"arg":"(t)pi","id":"Ryft"}],\n            [{"arg":"(-t)pi","id":"Rzft"}]\n        ]}\n    ', escape_url=False)
    with pytest.raises(ValueError, match='unsupported'):
        _ = circuit_to_quirk_url(cirq.Circuit(cirq.rx(sympy.FallingFactorial(t, t)).on(a)))