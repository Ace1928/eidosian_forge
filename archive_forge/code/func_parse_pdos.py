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
def parse_pdos(dos_file=None, spin_channel=None, total=False):
    """
    Parse a single DOS file created by cp2k. Must contain one PDOS snapshot. i.e. you cannot
    use this cannot deal with multiple concatenated dos files.

    Args:
        dos_file (list): list of pdos_ALPHA file paths
        spin_channel (int): Which spin channel the file corresponds to. By default, CP2K will
            write the file with ALPHA or BETA in the filename (for spin up or down), but
            you can specify this here, in case you have a manual file name.
            spin_channel == 1 --> spin up, spin_channel == -1 --> spin down.
        total (bool): Whether to grab the total occupations, or the orbital decomposed ones.
        sigma (float): width for gaussian smearing, if desired

    Returns:
        Everything necessary to create a dos object, in dict format:
            (1) orbital decomposed DOS dict:
                i.e. pdoss = {specie: {orbital.s: {Spin.up: ... }, orbital.px: {Spin.up: ... } ...}}
            (2) energy levels of this dos file
            (3) fermi energy (in eV).
        DOS object is not created here
    """
    spin = Spin(spin_channel) if spin_channel else Spin.down if 'BETA' in os.path.split(dos_file)[-1] else Spin.up
    with zopen(dos_file, mode='rt') as file:
        lines = file.readlines()
        kind = re.search('atomic kind\\s(.*)\\sat iter', lines[0]) or re.search('list\\s(\\d+)\\s(.*)\\sat iter', lines[0])
        kind = kind.groups()[0]
        header = re.split('\\s{2,}', lines[1].replace('#', '').strip())[2:]
        dat = np.loadtxt(dos_file)

        def cp2k_to_pmg_labels(label: str) -> str:
            if label == 'p':
                return 'px'
            if label == 'd':
                return 'dxy'
            if label == 'f':
                return 'f_3'
            if label == 'd-2':
                return 'dxy'
            if label == 'd-1':
                return 'dyz'
            if label == 'd0':
                return 'dz2'
            if label == 'd+1':
                return 'dxz'
            if label == 'd+2':
                return 'dx2'
            if label == 'f-3':
                return 'f_3'
            if label == 'f-2':
                return 'f_2'
            if label == 'f-1':
                return 'f_1'
            if label == 'f0':
                return 'f0'
            if label == 'f+1':
                return 'f1'
            if label == 'f+2':
                return 'f2'
            if label == 'f+3':
                return 'f3'
            return label
        header = [cp2k_to_pmg_labels(h) for h in header]
        data = np.delete(dat, 0, 1)
        occupations = data[:, 1]
        data = np.delete(data, 1, 1)
        data[:, 0] *= Ha_to_eV
        energies = data[:, 0]
        for idx, occu in enumerate(occupations):
            if occu == 0:
                break
            vbm_top = idx
        efermi = energies[vbm_top] + 1e-06
        energies = np.insert(energies, vbm_top + 1, np.linspace(energies[vbm_top] + 1e-06, energies[vbm_top + 1] - 1e-06, 2))
        data = np.insert(data, vbm_top + 1, np.zeros((2, data.shape[1])), axis=0)
        pdos = {kind: {getattr(Orbital, h): Dos(efermi=efermi, energies=energies, densities={spin: data[:, i + 1]}) for i, h in enumerate(header)}}
        if total:
            tdos = Dos(efermi=efermi, energies=energies, densities={spin: np.sum(data[:, 1:], axis=1)})
            return (pdos, tdos)
        return pdos