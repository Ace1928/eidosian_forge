import json
import numpy as np
import pytest
from ase.calculators.morse import MorsePotential
from ase.optimize import BFGS, ODE12r
from ase.optimize.precon import Exp
from ase.build import bulk
from ase.neb import NEB, NEBTools, NEBOptimizer
from ase.geometry.geometry import find_mic
from ase.constraints import FixBondLength
from ase.geometry.geometry import get_distances
from ase.utils.forcecurve import fit_images
@pytest.mark.parametrize('method', ['ODE', 'static'])
@pytest.mark.filterwarnings('ignore:NEBOptimizer did not converge')
def test_neb_optimizers(setup_images, method):
    images, _, _ = setup_images
    mep = NEB(images, method='spline', precon='Exp')
    mep.get_forces()
    R0 = mep.get_residual()
    opt = NEBOptimizer(mep, method=method)
    opt.run(steps=2)
    R1 = mep.get_residual()
    assert R1 < R0