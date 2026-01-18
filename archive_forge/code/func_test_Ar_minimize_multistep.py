import pytest
from numpy.testing import assert_allclose
from ase.cluster.icosahedron import Icosahedron
from ase.data import atomic_numbers, atomic_masses
from ase.optimize import LBFGS
@pytest.mark.calculator_lite
@pytest.mark.calculator('lammpsrun')
def test_Ar_minimize_multistep(factory, ar_nc, params):
    ar_nc = Icosahedron('Ar', noshells=2)
    ar_nc.cell = [[300, 0, 0], [0, 300, 0], [0, 0, 300]]
    ar_nc.pbc = True
    with factory.calc(specorder=['Ar'], **params) as calc:
        ar_nc.calc = calc
        F1_numer = calc.calculate_numerical_forces(ar_nc)
        assert_allclose(ar_nc.get_potential_energy(), -0.468147667942117, atol=0.0001, rtol=0.0001)
        assert_allclose(ar_nc.get_forces(), F1_numer, atol=0.0001, rtol=0.0001)
        params['minimize'] = '1.0e-15 1.0e-6 2000 4000'
        calc.parameters = params
        calc.run(set_atoms=True)
        ar_nc.set_positions(calc.atoms.positions)
        assert_allclose(ar_nc.get_potential_energy(), -0.4791815887032201, atol=0.0001, rtol=0.0001)
        assert_allclose(ar_nc.get_forces(), calc.calculate_numerical_forces(ar_nc), atol=0.0001, rtol=0.0001)