import collections
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer
def read_everything(self):
    dct = dict(self._read_everything())
    spinpol = dct.pop('spinpol')
    if spinpol:
        for name in ('eigenvalues', 'occupations'):
            array = dct[name]
            _, nkpts, nbands_double = array.shape
            assert _ == 1
            assert nbands_double % 2 == 0
            nbands = nbands_double // 2
            newarray = np.empty((2, nkpts, nbands))
            newarray[0, :, :] = array[0, :, :nbands]
            newarray[1, :, :] = array[0, :, nbands:]
            if name == 'eigenvalues':
                diffs = np.diff(newarray, axis=2)
                assert all(diffs.flat[:] > 0)
            dct[name] = newarray
    return dct