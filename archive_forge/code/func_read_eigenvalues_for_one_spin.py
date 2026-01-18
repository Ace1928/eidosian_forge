import os
from os.path import join
import re
from glob import glob
import warnings
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr, fs
from ase.calculators.calculator import Parameters
def read_eigenvalues_for_one_spin(fd, nkpts):
    kpoint_weights = []
    kpoint_coords = []
    eig_kn = []
    for ikpt in range(nkpts):
        header = next(fd)
        nbands, weight, kvector = match_kpt_header(header)
        kpoint_coords.append(kvector)
        kpoint_weights.append(weight)
        eig_n = []
        while len(eig_n) < nbands:
            line = next(fd)
            tokens = line.split()
            values = np.array(tokens).astype(float) * Hartree
            eig_n.extend(values)
        assert len(eig_n) == nbands
        eig_kn.append(eig_n)
        assert nbands == len(eig_kn[0])
    kpoint_weights = np.array(kpoint_weights)
    kpoint_coords = np.array(kpoint_coords)
    eig_kn = np.array(eig_kn)
    return (kpoint_coords, kpoint_weights, eig_kn)