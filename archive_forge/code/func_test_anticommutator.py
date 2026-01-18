from __future__ import annotations
from typing import Any
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.cg import CG, Wigner3j, Wigner6j, Wigner9j
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import CGate, CNotGate, IdentityGate, UGate, XGate
from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace, HilbertSpace, L2
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import Operator, OuterProduct, DifferentialOperator
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.qubit import Qubit, IntQubit
from sympy.physics.quantum.spin import Jz, J2, JzBra, JzBraCoupled, JzKet, JzKetCoupled, Rotation, WignerD
from sympy.physics.quantum.state import Bra, Ket, TimeDepBra, TimeDepKet
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.sho1d import RaisingOp
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices.dense import Matrix
from sympy.sets.sets import Interval
from sympy.testing.pytest import XFAIL
from sympy.physics.quantum.spin import JzOp
from sympy.printing import srepr
from sympy.printing.pretty import pretty as xpretty
from sympy.printing.latex import latex
def test_anticommutator():
    A = Operator('A')
    B = Operator('B')
    ac = AntiCommutator(A, B)
    ac_tall = AntiCommutator(A ** 2, B)
    assert str(ac) == '{A,B}'
    assert pretty(ac) == '{A,B}'
    assert upretty(ac) == '{A,B}'
    assert latex(ac) == '\\left\\{A,B\\right\\}'
    sT(ac, "AntiCommutator(Operator(Symbol('A')),Operator(Symbol('B')))")
    assert str(ac_tall) == '{A**2,B}'
    ascii_str = '/ 2  \\\n<A ,B>\n\\    /'
    ucode_str = '⎧ 2  ⎫\n⎨A ,B⎬\n⎩    ⎭'
    assert pretty(ac_tall) == ascii_str
    assert upretty(ac_tall) == ucode_str
    assert latex(ac_tall) == '\\left\\{A^{2},B\\right\\}'
    sT(ac_tall, "AntiCommutator(Pow(Operator(Symbol('A')), Integer(2)),Operator(Symbol('B')))")