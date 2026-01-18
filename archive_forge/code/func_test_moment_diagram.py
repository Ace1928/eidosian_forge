import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_moment_diagram():
    a, _, c, d = cirq.GridQubit.rect(2, 2)
    m = cirq.Moment(cirq.CZ(a, d), cirq.X(c).with_classical_controls('m'))
    assert str(m).strip() == '\n  ╷ 0                 1\n╶─┼─────────────────────\n0 │ @─────────────────┐\n  │                   │\n1 │ X(conditions=[m]) @\n  │\n    '.strip()