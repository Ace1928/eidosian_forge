import numpy as np
from ase import Atom
from ase.build import bulk
from ase.calculators.checkpoint import Checkpoint, CheckpointCalculator
from ase.calculators.lj import LennardJones
from ase.lattice.cubic import Diamond
def test_new_style_interface(testdir):
    calc = LennardJones()
    atoms = bulk('Cu')
    rattle_calc(atoms, calc)