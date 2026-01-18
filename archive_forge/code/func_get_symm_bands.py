from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def get_symm_bands(self, structure: Structure, efermi, kpt_line=None, labels_dict=None):
    """Function useful to read bands from Boltztrap output and get a
        BandStructureSymmLine object comparable with that one from a DFT
        calculation (if the same kpt_line is provided). Default kpt_line
        and labels_dict is the standard path of high symmetry k-point for
        the specified structure. They could be extracted from the
        BandStructureSymmLine object that you want to compare with. efermi
        variable must be specified to create the BandStructureSymmLine
        object (usually it comes from DFT or Boltztrap calc).
        """
    try:
        if kpt_line is None:
            kpath = HighSymmKpath(structure)
            kpt_line = [Kpoint(kpt, structure.lattice.reciprocal_lattice) for kpt in kpath.get_kpoints(coords_are_cartesian=False)[0]]
            labels_dict = {label: key for key, label in zip(*kpath.get_kpoints(coords_are_cartesian=False)) if label}
            kpt_line = [kp.frac_coords for kp in kpt_line]
        elif isinstance(kpt_line[0], Kpoint):
            kpt_line = [kp.frac_coords for kp in kpt_line]
            labels_dict = {k: labels_dict[k].frac_coords for k in labels_dict}
        _idx_list: list[tuple[int, ArrayLike]] = []
        for idx, kp in enumerate(kpt_line):
            w: list[bool] = []
            prec = 1e-05
            while len(w) == 0:
                w = np.where(np.all(np.abs(kp - self._bz_kpoints) < [prec] * 3, axis=1))[0]
                prec *= 10
            _idx_list.append((idx, w[0]))
        idx_list = np.array(_idx_list)
        bz_bands_in_eV = (self._bz_bands * Energy(1, 'Ry').to('eV') + efermi).T
        bands_dict = {Spin.up: bz_bands_in_eV[:, idx_list[:, 1]].tolist()}
        return BandStructureSymmLine(kpt_line, bands_dict, structure.lattice.reciprocal_lattice, efermi, labels_dict=labels_dict)
    except Exception:
        raise BoltztrapError('Bands are not in output of BoltzTraP.\nBolztrapRunner must be run with run_type=BANDS')