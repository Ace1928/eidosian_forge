import numpy as np
from ase import Atom
from ase.build import bulk
from ase.calculators.checkpoint import Checkpoint, CheckpointCalculator
from ase.calculators.lj import LennardJones
from ase.lattice.cubic import Diamond
def op2(a, m):
    a += Atom('C', m * np.array([0.2, 0.3, 0.1]))
    return (a, a.positions[0])