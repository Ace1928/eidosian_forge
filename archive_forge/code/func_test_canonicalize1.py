from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag
def test_canonicalize1():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Lorentz)
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(-d0) * A(d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0)*A(-L_0)'
    t = A(-d0) * A(-d1) * A(-d2) * A(d2) * A(d1) * A(d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0)*A(-L_0)*A(L_1)*A(-L_1)*A(L_2)*A(-L_2)'
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1), 1)
    t = A(-d0) * A(-d1) * A(-d2) * A(d2) * A(d1) * A(d0)
    tc = t.canon_bp()
    assert tc == 0
    A = TensorHead('A', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    t = A(d0, b) * A(a, -d1) * A(d1, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, L_0)*A(b, L_1)*A(-L_0, -L_1)'
    B = TensorHead('B', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    t = A(d0, b) * A(d1, -d0) * B(a, -d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(b, L_0)*A(-L_0, L_1)*B(a, -L_1)'
    A = TensorHead('A', [Lorentz] * 3, TensorSymmetry.fully_symmetric(3))
    t = A(d1, d0, b) * A(a, -d1, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, L_0, L_1)*A(b, -L_0, -L_1)'
    t = A(d3, d0, d2) * A(a0, -d1, -d2) * A(d1, -d3, a1) * A(a2, a3, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(a0, L_0, L_1)*A(a1, -L_0, L_2)*A(a2, a3, L_3)*A(-L_1, -L_2, -L_3)'
    A = TensorHead('A', [Lorentz] * 3, TensorSymmetry.fully_symmetric(3))
    B = TensorHead('B', [Lorentz] * 2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2) * A(-d2, -d3, -d1) * B(-d0, d3)
    tc = t.canon_bp()
    assert tc == 0
    A = TensorHead('A', [Lorentz] * 3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Lorentz] * 2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2) * A(-d2, -d3, -d1) * B(-d0, d3)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1, L_2)*A(-L_0, -L_1, L_3)*B(-L_2, -L_3)'
    Spinor = TensorIndexType('Spinor', dummy_name='S', metric_symmetry=-1)
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Spinor)
    A = TensorHead('A', [Spinor] * 3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Spinor] * 2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2) * A(-d2, -d3, -d1) * B(-d0, d3)
    tc = t.canon_bp()
    assert str(tc) == '-A(S_0, S_1, S_2)*A(-S_0, -S_1, S_3)*B(-S_2, -S_3)'
    Mat = TensorIndexType('Mat', metric_symmetry=0, dummy_name='M')
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Mat)
    A = TensorHead('A', [Mat] * 3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Mat] * 2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2) * A(-d2, -d3, -d1) * B(-d0, d3)
    tc = t.canon_bp()
    assert str(tc) == 'A(M_0, M_1, M_2)*A(-M_0, -M_1, -M_3)*B(-M_2, M_3)'
    alpha, beta, gamma, mu, nu, rho = tensor_indices('alpha,beta,gamma,mu,nu,rho', Lorentz)
    Gamma = TensorHead('Gamma', [Lorentz], TensorSymmetry.fully_symmetric(1), 2)
    Gamma2 = TensorHead('Gamma', [Lorentz] * 2, TensorSymmetry.fully_symmetric(-2), 2)
    Gamma3 = TensorHead('Gamma', [Lorentz] * 3, TensorSymmetry.fully_symmetric(-3), 2)
    t = Gamma2(-mu, -nu) * Gamma(rho) * Gamma3(nu, mu, alpha)
    tc = t.canon_bp()
    assert str(tc) == '-Gamma(L_0, L_1)*Gamma(rho)*Gamma(alpha, -L_0, -L_1)'
    t = Gamma2(mu, nu) * Gamma2(beta, gamma) * Gamma(-rho) * Gamma3(alpha, -mu, -nu)
    tc = t.canon_bp()
    assert str(tc) == 'Gamma(L_0, L_1)*Gamma(beta, gamma)*Gamma(-rho)*Gamma(alpha, -L_0, -L_1)'
    Flavor = TensorIndexType('Flavor', dummy_name='F')
    a, b, c, d, e, ff = tensor_indices('a,b,c,d,e,f', Flavor)
    mu, nu = tensor_indices('mu,nu', Lorentz)
    f = TensorHead('f', [Flavor] * 3, TensorSymmetry.direct_product(1, -2))
    A = TensorHead('A', [Lorentz, Flavor], TensorSymmetry.no_symmetry(2))
    t = f(c, -d, -a) * f(-c, -e, -b) * A(-mu, d) * A(-nu, a) * A(nu, e) * A(mu, b)
    tc = t.canon_bp()
    assert str(tc) == '-f(F_0, F_1, F_2)*f(-F_0, F_3, F_4)*A(L_0, -F_1)*A(-L_0, -F_3)*A(L_1, -F_2)*A(-L_1, -F_4)'