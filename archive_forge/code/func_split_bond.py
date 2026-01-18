from ase.atoms import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
def split_bond(atoms, index1, index2):
    """Split atoms by a bond specified by indices"""
    assert index1 != index2
    if index2 > index1:
        shift = (0, 1)
    else:
        shift = (1, 0)
    atoms_copy = atoms.copy()
    del atoms_copy[index2]
    atoms1 = connected_atoms(atoms_copy, index1 - shift[0])
    atoms_copy = atoms.copy()
    del atoms_copy[index1]
    atoms2 = connected_atoms(atoms_copy, index2 - shift[1])
    return (atoms1, atoms2)