import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT as OrigEMT
from ase.dyneb import DyNEB
from ase.optimize import BFGS
def run_NEB():
    if method == 'dyn':
        neb = DyNEB(images, fmax=fmax, dynamic_relaxation=True)
        neb.interpolate()
    elif method == 'dyn_scale':
        neb = DyNEB(images, fmax=fmax, dynamic_relaxation=True, scale_fmax=6.0)
        neb.interpolate()
    else:
        neb = DyNEB(images, dynamic_relaxation=False)
        neb.interpolate()
    force_evaluations[0] = 0
    opt = BFGS(neb)
    opt.run(fmax=fmax)
    force_calls.append(force_evaluations[0])
    Emax.append(np.sort([image.get_potential_energy() for image in images[1:-1]])[-1])