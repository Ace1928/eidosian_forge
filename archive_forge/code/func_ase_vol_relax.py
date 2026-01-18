import pytest
import numpy as np
from ase import io
from ase.optimize import BFGS
from ase.build import bulk
def ase_vol_relax():
    Al = bulk('Al', 'fcc', a=4.5, cubic=True)
    calc = factory.calc(xc='LDA')
    Al.calc = calc
    from ase.constraints import StrainFilter
    sf = StrainFilter(Al)
    with BFGS(sf, logfile='relaxation.log') as qn:
        qn.run(fmax=0.1, steps=5)
    print('Stress:\n', calc.read_stress())
    print('Al post ASE volume relaxation\n', calc.get_atoms().get_cell())
    return Al