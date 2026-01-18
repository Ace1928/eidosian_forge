from sympy.concrete.tests.test_sums_products import NS
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units import convert_to, coulomb_constant, elementary_charge, gravitational_constant, planck
from sympy.physics.units.definitions.unit_definitions import angstrom, statcoulomb, coulomb, second, gram, centimeter, erg, \
from sympy.physics.units.systems import SI
from sympy.physics.units.systems.cgs import cgs_gauss
def test_conversion_to_from_si():
    assert convert_to(statcoulomb, coulomb, cgs_gauss) == coulomb / 2997924580
    assert convert_to(coulomb, statcoulomb, cgs_gauss) == 2997924580 * statcoulomb
    assert convert_to(statcoulomb, sqrt(gram * centimeter ** 3) / second, cgs_gauss) == centimeter ** (S(3) / 2) * sqrt(gram) / second
    assert convert_to(coulomb, sqrt(gram * centimeter ** 3) / second, cgs_gauss) == 2997924580 * centimeter ** (S(3) / 2) * sqrt(gram) / second
    assert convert_to(coulomb, statcoulomb, SI) == coulomb
    assert convert_to(statcoulomb, coulomb, SI) == statcoulomb
    assert convert_to(erg, joule, SI) == joule / 10 ** 7
    assert convert_to(erg, joule, cgs_gauss) == joule / 10 ** 7
    assert convert_to(joule, erg, SI) == 10 ** 7 * erg
    assert convert_to(joule, erg, cgs_gauss) == 10 ** 7 * erg
    assert convert_to(dyne, newton, SI) == newton / 10 ** 5
    assert convert_to(dyne, newton, cgs_gauss) == newton / 10 ** 5
    assert convert_to(newton, dyne, SI) == 10 ** 5 * dyne
    assert convert_to(newton, dyne, cgs_gauss) == 10 ** 5 * dyne