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
def parse_dos(self, dos_file=None, pdos_files=None, ldos_files=None):
    """
        Parse the dos files produced by cp2k calculation. CP2K produces different files based
        on the input file rather than assimilating them all into one file.

        One file type is the overall DOS file, which is used for k-point calculations. For
        non-kpoint calculation, the overall DOS is generally not calculated, but the
        element-projected pDOS is. Separate files are created for each spin channel and each
        atom kind. If requested, cp2k can also do site/local projected dos (ldos). Each site
        requested will have a separate file for each spin channel (if spin polarized calculation
        is performed).

        If possible, this function will assimilate the ldos files into a CompleteDos object.
        Either provide a list of PDOS file paths, or use glob to find the .pdos_ALPHA extension
        in the calculation directory.

        Args:
            dos_file (str): Name of the dos file, otherwise will be inferred
            pdos_files (list): list of pdos file paths, otherwise they will be inferred
            ldos_files (list): list of ldos file paths, otherwise they will be inferred
        """
    if dos_file is None:
        dos_file = self.filenames['DOS'][0] if self.filenames['DOS'] else None
    if pdos_files is None:
        pdos_files = self.filenames['PDOS']
    if ldos_files is None:
        ldos_files = self.filenames['LDOS']
    tdos, pdoss, ldoss = (None, {}, {})
    for pdos_file in pdos_files:
        _pdos, _tdos = parse_pdos(pdos_file, total=True)
        for k in _pdos:
            if k in pdoss:
                for orbital in _pdos[k]:
                    pdoss[k][orbital].densities.update(_pdos[k][orbital].densities)
            else:
                pdoss.update(_pdos)
        if not tdos:
            tdos = _tdos
        else:
            for k, v in _tdos.densities.copy().items():
                if k not in tdos.densities:
                    tdos.densities[Spin(int(k))] = [0] * len(v)
                tdos.densities[k] = np.array(tdos.densities[k]) + np.array(_tdos.densities[k])
    for ldos_file in ldos_files:
        _pdos = parse_pdos(ldos_file)
        for k in _pdos:
            if k in ldoss:
                for orbital in _pdos[k]:
                    ldoss[k][orbital].densities.update(_pdos[k][orbital].densities)
            else:
                ldoss.update(_pdos)
    self.data['pdos'] = jsanitize(pdoss, strict=True)
    self.data['ldos'] = jsanitize(ldoss, strict=True)
    self.data['tdos'] = parse_dos(dos_file) if dos_file else tdos
    if self.data.get('tdos'):
        self.band_gap = self.data['tdos'].get_gap()
        self.cbm, self.vbm = self.data['tdos'].get_cbm_vbm()
        self.efermi = (self.cbm + self.vbm) / 2
    _ldoss = {}
    if self.initial_structure and len(ldoss) == len(self.initial_structure):
        for k, lds in ldoss.items():
            _ldoss[self.initial_structure[int(k) - 1]] = {Orbital(orb): lds[orb].densities for orb in lds}
        self.data['cdos'] = CompleteDos(self.final_structure, total_dos=tdos, pdoss=_ldoss)