import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def process_special_kwargs(atoms, kwargs):
    kwargs = kwargs.copy()
    kpts = kwargs.pop('kpts', None)
    if kpts is not None:
        for kw in ['kpoints', 'reducedkpoints', 'kpointsgrid']:
            if kw in kwargs:
                raise ValueError('k-points specified multiple times')
        kptsarray = kpts2ndarray(kpts, atoms)
        nkpts = len(kptsarray)
        fullarray = np.empty((nkpts, 4))
        fullarray[:, 0] = 1.0 / nkpts
        fullarray[:, 1:4] = kptsarray
        kwargs['kpointsreduced'] = fullarray.tolist()
    for kw in special_ase_keywords:
        assert kw not in kwargs, kw
    return kwargs