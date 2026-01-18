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
def test_innerproduct():
    x = symbols('x')
    ip1 = InnerProduct(Bra(), Ket())
    ip2 = InnerProduct(TimeDepBra(), TimeDepKet())
    ip3 = InnerProduct(JzBra(1, 1), JzKet(1, 1))
    ip4 = InnerProduct(JzBraCoupled(1, 1, (1, 1)), JzKetCoupled(1, 1, (1, 1)))
    ip_tall1 = InnerProduct(Bra(x / 2), Ket(x / 2))
    ip_tall2 = InnerProduct(Bra(x), Ket(x / 2))
    ip_tall3 = InnerProduct(Bra(x / 2), Ket(x))
    assert str(ip1) == '<psi|psi>'
    assert pretty(ip1) == '<psi|psi>'
    assert upretty(ip1) == '⟨ψ❘ψ⟩'
    assert latex(ip1) == '\\left\\langle \\psi \\right. {\\left|\\psi\\right\\rangle }'
    sT(ip1, "InnerProduct(Bra(Symbol('psi')),Ket(Symbol('psi')))")
    assert str(ip2) == '<psi;t|psi;t>'
    assert pretty(ip2) == '<psi;t|psi;t>'
    assert upretty(ip2) == '⟨ψ;t❘ψ;t⟩'
    assert latex(ip2) == '\\left\\langle \\psi;t \\right. {\\left|\\psi;t\\right\\rangle }'
    sT(ip2, "InnerProduct(TimeDepBra(Symbol('psi'),Symbol('t')),TimeDepKet(Symbol('psi'),Symbol('t')))")
    assert str(ip3) == '<1,1|1,1>'
    assert pretty(ip3) == '<1,1|1,1>'
    assert upretty(ip3) == '⟨1,1❘1,1⟩'
    assert latex(ip3) == '\\left\\langle 1,1 \\right. {\\left|1,1\\right\\rangle }'
    sT(ip3, 'InnerProduct(JzBra(Integer(1),Integer(1)),JzKet(Integer(1),Integer(1)))')
    assert str(ip4) == '<1,1,j1=1,j2=1|1,1,j1=1,j2=1>'
    assert pretty(ip4) == '<1,1,j1=1,j2=1|1,1,j1=1,j2=1>'
    assert upretty(ip4) == '⟨1,1,j₁=1,j₂=1❘1,1,j₁=1,j₂=1⟩'
    assert latex(ip4) == '\\left\\langle 1,1,j_{1}=1,j_{2}=1 \\right. {\\left|1,1,j_{1}=1,j_{2}=1\\right\\rangle }'
    sT(ip4, 'InnerProduct(JzBraCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))),JzKetCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))))')
    assert str(ip_tall1) == '<x/2|x/2>'
    ascii_str = ' / | \\ \n/ x|x \\\n\\ -|- /\n \\2|2/ '
    ucode_str = ' ╱ │ ╲ \n╱ x│x ╲\n╲ ─│─ ╱\n ╲2│2╱ '
    assert pretty(ip_tall1) == ascii_str
    assert upretty(ip_tall1) == ucode_str
    assert latex(ip_tall1) == '\\left\\langle \\frac{x}{2} \\right. {\\left|\\frac{x}{2}\\right\\rangle }'
    sT(ip_tall1, "InnerProduct(Bra(Mul(Rational(1, 2), Symbol('x'))),Ket(Mul(Rational(1, 2), Symbol('x'))))")
    assert str(ip_tall2) == '<x|x/2>'
    ascii_str = ' / | \\ \n/  |x \\\n\\ x|- /\n \\ |2/ '
    ucode_str = ' ╱ │ ╲ \n╱  │x ╲\n╲ x│─ ╱\n ╲ │2╱ '
    assert pretty(ip_tall2) == ascii_str
    assert upretty(ip_tall2) == ucode_str
    assert latex(ip_tall2) == '\\left\\langle x \\right. {\\left|\\frac{x}{2}\\right\\rangle }'
    sT(ip_tall2, "InnerProduct(Bra(Symbol('x')),Ket(Mul(Rational(1, 2), Symbol('x'))))")
    assert str(ip_tall3) == '<x/2|x>'
    ascii_str = ' / | \\ \n/ x|  \\\n\\ -|x /\n \\2| / '
    ucode_str = ' ╱ │ ╲ \n╱ x│  ╲\n╲ ─│x ╱\n ╲2│ ╱ '
    assert pretty(ip_tall3) == ascii_str
    assert upretty(ip_tall3) == ucode_str
    assert latex(ip_tall3) == '\\left\\langle \\frac{x}{2} \\right. {\\left|x\\right\\rangle }'
    sT(ip_tall3, "InnerProduct(Bra(Mul(Rational(1, 2), Symbol('x'))),Ket(Symbol('x')))")