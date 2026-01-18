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
@staticmethod
def check_acc_bzt_bands(sbs_bz, sbs_ref, warn_thr=(0.03, 0.03)):
    """Compare sbs_bz BandStructureSymmLine calculated with boltztrap with
        the sbs_ref BandStructureSymmLine as reference (from MP for
        instance), computing correlation and energy difference for eight bands
        around the gap (semiconductors) or Fermi level (metals).
        warn_thr is a threshold to get a warning in the accuracy of Boltztap
        interpolated bands.

        Return a dictionary with these keys:
        - "N": the index of the band compared; inside each there are:
            - "Corr": correlation coefficient for the 8 compared bands
            - "Dist": energy distance for the 8 compared bands
            - "branch_name": energy distance for that branch
        - "avg_corr": average of correlation coefficient over the 8 bands
        - "avg_dist": average of energy distance over the 8 bands
        - "nb_list": list of indexes of the 8 compared bands
        - "acc_thr": list of two float corresponding to the two warning thresholds in input
        - "acc_err": list of two bools:
            True if the avg_corr > warn_thr[0], and
            True if the avg_dist > warn_thr[1]
        See also compare_sym_bands function doc.
        """
    if not sbs_ref.is_metal() and (not sbs_bz.is_metal()):
        vbm_idx = sbs_bz.get_vbm()['band_index'][Spin.up][-1]
        cbm_idx = sbs_bz.get_cbm()['band_index'][Spin.up][0]
        nb_list = range(vbm_idx - 3, cbm_idx + 4)
    else:
        bnd_around_efermi = []
        delta = 0
        spin = next(iter(sbs_bz.bands))
        while len(bnd_around_efermi) < 8 and delta < 100:
            delta += 0.1
            bnd_around_efermi = []
            for nb in range(len(sbs_bz.bands[spin])):
                for kp in range(len(sbs_bz.bands[spin][nb])):
                    if abs(sbs_bz.bands[spin][nb][kp] - sbs_bz.efermi) < delta:
                        bnd_around_efermi.append(nb)
                        break
        if len(bnd_around_efermi) < 8:
            print(f'Warning! check performed on {len(bnd_around_efermi)}')
            nb_list = bnd_around_efermi
        else:
            nb_list = bnd_around_efermi[:8]
    bcheck = compare_sym_bands(sbs_bz, sbs_ref, nb_list)
    acc_err = [False, False]
    avg_corr = sum((item[1]['Corr'] for item in bcheck.items())) / 8
    avg_distance = sum((item[1]['Dist'] for item in bcheck.items())) / 8
    if avg_corr > warn_thr[0]:
        acc_err[0] = True
    if avg_distance > warn_thr[0]:
        acc_err[1] = True
    bcheck['avg_corr'] = avg_corr
    bcheck['avg_distance'] = avg_distance
    bcheck['acc_err'] = acc_err
    bcheck['acc_thr'] = warn_thr
    bcheck['nb_list'] = nb_list
    if True in acc_err:
        print('Warning! some bands around gap are not accurate')
    return bcheck