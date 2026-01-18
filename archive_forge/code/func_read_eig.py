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
def read_eig(fd):
    line = next(fd)
    results = {}
    m = re.match('\\s*Fermi \\(or HOMO\\) energy \\(hartree\\)\\s*=\\s*(\\S+)', line)
    if m is not None:
        results['fermilevel'] = float(m.group(1)) * Hartree
        line = next(fd)
    nspins = 1
    m = re.match('\\s*Magnetization \\(Bohr magneton\\)=\\s*(\\S+)', line)
    if m is not None:
        nspins = 2
        magmom = float(m.group(1))
        results['magmom'] = magmom
        line = next(fd)
    if 'Total spin up' in line:
        assert nspins == 2
        line = next(fd)
    m = re.match('\\s*Eigenvalues \\(hartree\\) for nkpt\\s*=\\s*(\\S+)\\s*k\\s*points', line)
    if 'SPIN' in line or 'spin' in line:
        nspins = 2
    assert m is not None
    nkpts = int(m.group(1))
    eig_skn = []
    kpts, weights, eig_kn = read_eigenvalues_for_one_spin(fd, nkpts)
    nbands = eig_kn.shape[1]
    eig_skn.append(eig_kn)
    if nspins == 2:
        line = next(fd)
        assert 'SPIN DOWN' in line
        _, _, eig_kn = read_eigenvalues_for_one_spin(fd, nkpts)
        assert eig_kn.shape == (nkpts, nbands)
        eig_skn.append(eig_kn)
    eig_skn = np.array(eig_skn)
    eigshape = (nspins, nkpts, nbands)
    assert eig_skn.shape == eigshape, (eig_skn.shape, eigshape)
    results['ibz_kpoints'] = kpts
    results['kpoint_weights'] = weights
    results['eigenvalues'] = eig_skn
    return results