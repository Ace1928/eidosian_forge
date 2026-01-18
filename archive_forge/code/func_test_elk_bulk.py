import pytest
from ase.build import bulk
@pytest.mark.calculator_lite
@pytest.mark.parametrize('atoms', systems(), ids=lambda atoms: str(atoms.symbols))
@pytest.mark.calculator('elk', tasks=0, ngridk=(3, 3, 3))
def test_elk_bulk(factory, atoms):
    calc = factory.calc()
    atoms.calc = calc
    spinpol = atoms.get_initial_magnetic_moments().any()
    props = atoms.get_properties(['energy', 'forces'])
    energy = props['energy']
    if str(atoms.symbols) == 'Si2':
        assert energy == pytest.approx(-15729.719246, abs=0.1)
        assert atoms.get_potential_energy() == pytest.approx(energy)
    expected_props = {'energy', 'free_energy', 'forces', 'ibz_kpoints', 'eigenvalues', 'occupations'}
    assert expected_props < set(props)
    assert calc.get_fermi_level() == props['fermi_level']
    assert calc.get_ibz_k_points() == pytest.approx(props['ibz_kpoints'])
    assert calc.get_k_point_weights() == pytest.approx(props['kpoint_weights'])
    I = slice(None)
    assert calc.get_eigenvalues(I, I) == pytest.approx(props['eigenvalues'])
    assert calc.get_occupation_numbers(I, I) == pytest.approx(props['occupations'])
    assert calc.get_spin_polarized() == spinpol
    assert calc.get_number_of_spins() == 1 + int(spinpol)
    assert calc.get_number_of_bands() == props['nbands']