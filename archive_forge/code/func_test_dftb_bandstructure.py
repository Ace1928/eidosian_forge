import pytest
from ase.build import bulk
from ase.utils import tokenize_version
@pytest.mark.skip('test is rather broken')
def test_dftb_bandstructure(dftb_factory):
    version = dftb_factory.version()
    if tokenize_version(version) < tokenize_version('17.1'):
        pytest.skip('Band structure requires DFTB 17.1+')
    calc = dftb_factory.calc(label='dftb', kpts=(3, 3, 3), Hamiltonian_SCC='Yes', Hamiltonian_SCCTolerance=1e-05, Hamiltonian_MaxAngularMomentum_Si='d')
    atoms = bulk('Si')
    atoms.calc = calc
    atoms.get_potential_energy()
    efermi = calc.get_fermi_level()
    assert abs(efermi - -2.90086680996455) < 1.0
    calc = dftb_factory.calc(atoms=atoms, label='dftb', kpts={'path': 'WGXWLG', 'npoints': 50}, Hamiltonian_SCC='Yes', Hamiltonian_MaxSCCIterations=1, Hamiltonian_ReadInitialCharges='Yes', Hamiltonian_MaxAngularMomentum_Si='d')
    atoms.calc = calc
    calc.calculate(atoms)
    calc.band_structure()