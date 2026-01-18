from sympy.core.backend import symbols, Matrix, atan, zeros
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, Particle, Point,
from sympy.testing.pytest import raises
def test_form_2():
    symsystem2 = SymbolicSystem(coordinates, comb_implicit_rhs, speeds=speeds, mass_matrix=comb_implicit_mat, alg_con=alg_con_full, output_eqns=out_eqns, bodies=bodies, loads=loads)
    assert symsystem2.coordinates == Matrix([x, y, lam])
    assert symsystem2.speeds == Matrix([u, v])
    assert symsystem2.states == Matrix([x, y, lam, u, v])
    assert symsystem2.alg_con == [4]
    inter = comb_implicit_rhs
    assert simplify(symsystem2.comb_implicit_rhs - inter) == zeros(5, 1)
    assert simplify(symsystem2.comb_implicit_mat - comb_implicit_mat) == zeros(5)
    assert set(symsystem2.dynamic_symbols()) == {y, v, lam, u, x}
    assert type(symsystem2.dynamic_symbols()) == tuple
    assert set(symsystem2.constant_symbols()) == {l, g, m}
    assert type(symsystem2.constant_symbols()) == tuple
    inter = comb_explicit_rhs
    symsystem2.compute_explicit_form()
    assert simplify(symsystem2.comb_explicit_rhs - inter) == zeros(5, 1)
    assert symsystem2.output_eqns == out_eqns
    assert symsystem2.bodies == (Pa,)
    assert symsystem2.loads == ((P, g * m * N.x),)