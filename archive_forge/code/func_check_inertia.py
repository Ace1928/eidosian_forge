from ase.units import fs
from ase.build import bulk
from ase.md import Langevin
from ase.md.fix import FixRotation
from ase.utils import seterr
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
import numpy as np
def check_inertia(atoms):
    m, v = atoms.get_moments_of_inertia(vectors=True)
    print('Moments of inertia:')
    print(m)
    n = 0
    delta = 0.01
    for a in v:
        if abs(a[0]) < delta and abs(a[1]) < delta and (abs(abs(a[2]) - 1.0) < delta):
            print('Vector along z:', a)
            n += 1
        else:
            print('Vector not along z:', a)
    assert n == 1