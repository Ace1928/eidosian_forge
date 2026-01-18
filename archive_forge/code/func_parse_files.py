from __future__ import annotations
import logging
import os
import re
import warnings
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from monty.re import regrep
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys, postprocessor
from pymatgen.io.xyz import XYZ
def parse_files(self):
    """
        Identify files present in the directory with the cp2k output file. Looks for trajectories,
        dos, and cubes.
        """
    self.filenames['DOS'] = glob(os.path.join(self.dir, '*.dos*'))
    pdos = glob(os.path.join(self.dir, '*pdos*'))
    self.filenames['PDOS'] = []
    self.filenames['LDOS'] = []
    for p in pdos:
        if 'list' in p.split('/')[-1]:
            self.filenames['LDOS'].append(p)
        else:
            self.filenames['PDOS'].append(p)
    self.filenames['band_structure'] = glob(os.path.join(self.dir, '*BAND.bs*'))
    self.filenames['trajectory'] = glob(os.path.join(self.dir, '*pos*.xyz*'))
    self.filenames['forces'] = glob(os.path.join(self.dir, '*frc*.xyz*'))
    self.filenames['stress'] = glob(os.path.join(self.dir, '*stress*'))
    self.filenames['cell'] = glob(os.path.join(self.dir, '*.cell*'))
    self.filenames['ener'] = glob(os.path.join(self.dir, '*.ener*'))
    self.filenames['electron_density'] = glob(os.path.join(self.dir, '*ELECTRON_DENSITY*.cube*'))
    self.filenames['spin_density'] = glob(os.path.join(self.dir, '*SPIN_DENSITY*.cube*'))
    self.filenames['v_hartree'] = glob(os.path.join(self.dir, '*hartree*.cube*'))
    self.filenames['hyperfine_tensor'] = glob(os.path.join(self.dir, '*HYPERFINE*eprhyp*'))
    self.filenames['g_tensor'] = glob(os.path.join(self.dir, '*GTENSOR*data*'))
    self.filenames['spinspin_tensor'] = glob(os.path.join(self.dir, '*K*data*'))
    self.filenames['chi_tensor'] = glob(os.path.join(self.dir, '*CHI*data*'))
    self.filenames['nmr_shift'] = glob(os.path.join(self.dir, '*SHIFT*data*'))
    self.filenames['raman'] = glob(os.path.join(self.dir, '*raman*data*'))
    restart = glob(os.path.join(self.dir, '*restart*'))
    self.filenames['restart.bak'] = []
    self.filenames['restart'] = []
    for r in restart:
        if 'bak' in r.split('/')[-1]:
            self.filenames['restart.bak'].append(r)
        else:
            self.filenames['restart'].append(r)
    wfn = glob(os.path.join(self.dir, '*.wfn*')) + glob(os.path.join(self.dir, '*.kp*'))
    self.filenames['wfn.bak'] = []
    for w in wfn:
        if 'bak' in w.split('/')[-1]:
            self.filenames['wfn.bak'].append(w)
        else:
            self.filenames['wfn'] = w
    for filename in self.filenames.values():
        if hasattr(filename, 'sort'):
            filename.sort(key=natural_keys)