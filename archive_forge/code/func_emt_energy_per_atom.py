import pytest
from ase import Atoms
from ase.lattice import all_variants
from ase.build.supercells import make_supercell
from ase.calculators.emt import EMT
def emt_energy_per_atom(atoms):
    atoms = atoms.copy()
    atoms.calc = EMT()
    return atoms.get_potential_energy() / len(atoms)