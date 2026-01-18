def test_compare_atoms():
    """
    Check that Atoms.compare_atoms correctly accounts for the different
    types of system changes
    """
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import compare_atoms
    cell1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cell2 = cell1 * 2
    atoms1 = Atoms(cell=cell1)
    atoms2 = Atoms(cell=cell2)
    assert set(compare_atoms(atoms1, atoms2)) == {'cell'}
    atoms1 = Atoms()
    atoms2 = Atoms(numbers=[0], positions=[[0, 0, 0]])
    assert set(compare_atoms(atoms1, atoms2)) == {'positions', 'numbers'}
    atoms1 = Atoms(numbers=[0], positions=[[0, 0, 0]])
    atoms2 = Atoms(numbers=[0], positions=[[1, 0, 0]])
    assert set(compare_atoms(atoms1, atoms2)) == {'positions'}
    assert set(compare_atoms(atoms1, atoms2, excluded_properties={'positions'})) == set()
    atoms1 = Atoms(numbers=[0], positions=[[0, 0, 0]])
    atoms2 = Atoms(numbers=[0], positions=[[0, 0, 0]], charges=[1.13])
    assert set(compare_atoms(atoms1, atoms2)) == {'initial_charges'}