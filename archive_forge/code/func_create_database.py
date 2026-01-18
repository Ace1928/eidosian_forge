from math import sin, cos, pi
from ase import Atoms
from ase.build import fcc111, fcc100, add_adsorbate
from ase.db import connect
from ase.constraints import FixAtoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.cluster import wulff_construction
def create_database():
    db = connect('systems.db', append=False)
    for atoms, description in systems:
        name = atoms.get_chemical_formula()
        db.write(atoms, description=description, name=name)
    if False:
        for atoms, description in systems:
            for seed in range(5):
                a = atoms.copy()
                a.rattle(0.1, seed=seed)
                name = a.get_chemical_formula() + '-' + str(seed)
                db.write(a, description=description, seed=seed, name=name)