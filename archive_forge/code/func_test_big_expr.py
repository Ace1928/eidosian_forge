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
def test_big_expr():
    f = Function('f')
    x = symbols('x')
    e1 = Dagger(AntiCommutator(Operator('A') + Operator('B'), Pow(DifferentialOperator(Derivative(f(x), x), f(x)), 3)) * TensorProduct(Jz ** 2, Operator('A') + Operator('B'))) * (JzBra(1, 0) + JzBra(1, 1)) * (JzKet(0, 0) + JzKet(1, -1))
    e2 = Commutator(Jz ** 2, Operator('A') + Operator('B')) * AntiCommutator(Dagger(Operator('C') * Operator('D')), Operator('E').inv() ** 2) * Dagger(Commutator(Jz, J2))
    e3 = Wigner3j(1, 2, 3, 4, 5, 6) * TensorProduct(Commutator(Operator('A') + Dagger(Operator('B')), Operator('C') + Operator('D')), Jz - J2) * Dagger(OuterProduct(Dagger(JzBra(1, 1)), JzBra(1, 0))) * TensorProduct(JzKetCoupled(1, 1, (1, 1)) + JzKetCoupled(1, 0, (1, 1)), JzKetCoupled(1, -1, (1, 1)))
    e4 = (ComplexSpace(1) * ComplexSpace(2) + FockSpace() ** 2) * (L2(Interval(0, oo)) + HilbertSpace())
    assert str(e1) == '(Jz**2)x(Dagger(A) + Dagger(B))*{Dagger(DifferentialOperator(Derivative(f(x), x),f(x)))**3,Dagger(A) + Dagger(B)}*(<1,0| + <1,1|)*(|0,0> + |1,-1>)'
    ascii_str = '                 /                                      3        \\                                 \n                 |/                                   +\\         |                                 \n    2  / +    +\\ <|                    /d            \\ |   +    +>                                 \n/J \\ x \\A  + B /*||DifferentialOperator|--(f(x)),f(x)| | ,A  + B |*(<1,0| + <1,1|)*(|0,0> + |1,-1>)\n\\ z/             \\\\                    \\dx           / /         /                                 '
    ucode_str = '                 ⎧                                      3        ⎫                                 \n                 ⎪⎛                                   †⎞         ⎪                                 \n    2  ⎛ †    †⎞ ⎨⎜                    ⎛d            ⎞ ⎟   †    †⎬                                 \n⎛J ⎞ ⨂ ⎝A  + B ⎠⋅⎪⎜DifferentialOperator⎜──(f(x)),f(x)⎟ ⎟ ,A  + B ⎪⋅(⟨1,0❘ + ⟨1,1❘)⋅(❘0,0⟩ + ❘1,-1⟩)\n⎝ z⎠             ⎩⎝                    ⎝dx           ⎠ ⎠         ⎭                                 '
    assert pretty(e1) == ascii_str
    assert upretty(e1) == ucode_str
    assert latex(e1) == '{J_z^{2}}\\otimes \\left({A^{\\dagger} + B^{\\dagger}}\\right) \\left\\{\\left(DifferentialOperator\\left(\\frac{d}{d x} f{\\left(x \\right)},f{\\left(x \\right)}\\right)^{\\dagger}\\right)^{3},A^{\\dagger} + B^{\\dagger}\\right\\} \\left({\\left\\langle 1,0\\right|} + {\\left\\langle 1,1\\right|}\\right) \\left({\\left|0,0\\right\\rangle } + {\\left|1,-1\\right\\rangle }\\right)'
    sT(e1, "Mul(TensorProduct(Pow(JzOp(Symbol('J')), Integer(2)), Add(Dagger(Operator(Symbol('A'))), Dagger(Operator(Symbol('B'))))), AntiCommutator(Pow(Dagger(DifferentialOperator(Derivative(Function('f')(Symbol('x')), Tuple(Symbol('x'), Integer(1))),Function('f')(Symbol('x')))), Integer(3)),Add(Dagger(Operator(Symbol('A'))), Dagger(Operator(Symbol('B'))))), Add(JzBra(Integer(1),Integer(0)), JzBra(Integer(1),Integer(1))), Add(JzKet(Integer(0),Integer(0)), JzKet(Integer(1),Integer(-1))))")
    assert str(e2) == '[Jz**2,A + B]*{E**(-2),Dagger(D)*Dagger(C)}*[J2,Jz]'
    ascii_str = '[    2      ] / -2  +  +\\ [ 2   ]\n[/J \\ ,A + B]*<E  ,D *C >*[J ,J ]\n[\\ z/       ] \\         / [    z]'
    ucode_str = '⎡    2      ⎤ ⎧ -2  †  †⎫ ⎡ 2   ⎤\n⎢⎛J ⎞ ,A + B⎥⋅⎨E  ,D ⋅C ⎬⋅⎢J ,J ⎥\n⎣⎝ z⎠       ⎦ ⎩         ⎭ ⎣    z⎦'
    assert pretty(e2) == ascii_str
    assert upretty(e2) == ucode_str
    assert latex(e2) == '\\left[J_z^{2},A + B\\right] \\left\\{E^{-2},D^{\\dagger} C^{\\dagger}\\right\\} \\left[J^2,J_z\\right]'
    sT(e2, "Mul(Commutator(Pow(JzOp(Symbol('J')), Integer(2)),Add(Operator(Symbol('A')), Operator(Symbol('B')))), AntiCommutator(Pow(Operator(Symbol('E')), Integer(-2)),Mul(Dagger(Operator(Symbol('D'))), Dagger(Operator(Symbol('C'))))), Commutator(J2Op(Symbol('J')),JzOp(Symbol('J'))))")
    assert str(e3) == 'Wigner3j(1, 2, 3, 4, 5, 6)*[Dagger(B) + A,C + D]x(-J2 + Jz)*|1,0><1,1|*(|1,0,j1=1,j2=1> + |1,1,j1=1,j2=1>)x|1,-1,j1=1,j2=1>'
    ascii_str = '          [ +          ]  /   2     \\                                                                 \n/1  3  5\\*[B  + A,C + D]x |- J  + J |*|1,0><1,1|*(|1,0,j1=1,j2=1> + |1,1,j1=1,j2=1>)x |1,-1,j1=1,j2=1>\n|       |                 \\        z/                                                                 \n\\2  4  6/                                                                                             '
    ucode_str = '          ⎡ †          ⎤  ⎛   2     ⎞                                                                 \n⎛1  3  5⎞⋅⎣B  + A,C + D⎦⨂ ⎜- J  + J ⎟⋅❘1,0⟩⟨1,1❘⋅(❘1,0,j₁=1,j₂=1⟩ + ❘1,1,j₁=1,j₂=1⟩)⨂ ❘1,-1,j₁=1,j₂=1⟩\n⎜       ⎟                 ⎝        z⎠                                                                 \n⎝2  4  6⎠                                                                                             '
    assert pretty(e3) == ascii_str
    assert upretty(e3) == ucode_str
    assert latex(e3) == '\\left(\\begin{array}{ccc} 1 & 3 & 5 \\\\ 2 & 4 & 6 \\end{array}\\right) {\\left[B^{\\dagger} + A,C + D\\right]}\\otimes \\left({- J^2 + J_z}\\right) {\\left|1,0\\right\\rangle }{\\left\\langle 1,1\\right|} \\left({{\\left|1,0,j_{1}=1,j_{2}=1\\right\\rangle } + {\\left|1,1,j_{1}=1,j_{2}=1\\right\\rangle }}\\right)\\otimes {{\\left|1,-1,j_{1}=1,j_{2}=1\\right\\rangle }}'
    sT(e3, "Mul(Wigner3j(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6)), TensorProduct(Commutator(Add(Dagger(Operator(Symbol('B'))), Operator(Symbol('A'))),Add(Operator(Symbol('C')), Operator(Symbol('D')))), Add(Mul(Integer(-1), J2Op(Symbol('J'))), JzOp(Symbol('J')))), OuterProduct(JzKet(Integer(1),Integer(0)),JzBra(Integer(1),Integer(1))), TensorProduct(Add(JzKetCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))), JzKetCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))), JzKetCoupled(Integer(1),Integer(-1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))))")
    assert str(e4) == '(C(1)*C(2)+F**2)*(L2(Interval(0, oo))+H)'
    ascii_str = '// 1    2\\    x2\\   / 2    \\\n\\\\C  x C / + F  / x \\L  + H/'
    ucode_str = '⎛⎛ 1    2⎞    ⨂2⎞   ⎛ 2    ⎞\n⎝⎝C  ⨂ C ⎠ ⊕ F  ⎠ ⨂ ⎝L  ⊕ H⎠'
    assert pretty(e4) == ascii_str
    assert upretty(e4) == ucode_str
    assert latex(e4) == '\\left(\\left(\\mathcal{C}^{1}\\otimes \\mathcal{C}^{2}\\right)\\oplus {\\mathcal{F}}^{\\otimes 2}\\right)\\otimes \\left({\\mathcal{L}^2}\\left( \\left[0, \\infty\\right) \\right)\\oplus \\mathcal{H}\\right)'
    sT(e4, 'TensorProductHilbertSpace((DirectSumHilbertSpace(TensorProductHilbertSpace(ComplexSpace(Integer(1)),ComplexSpace(Integer(2))),TensorPowerHilbertSpace(FockSpace(),Integer(2)))),(DirectSumHilbertSpace(L2(Interval(Integer(0), oo, false, true)),HilbertSpace())))')