from ase.build import bulk
from ase.calculators.emt import EMT
def test_calc_properties():
    atoms = bulk('Au', cubic=True)
    atoms.calc = EMT()
    props = atoms.get_properties(['energy', 'stress', 'forces'])
    natoms = len(atoms)
    assert props['stress'].shape == (6,)
    assert props['forces'].shape == (natoms, 3)
    assert isinstance(props['energy'], float)