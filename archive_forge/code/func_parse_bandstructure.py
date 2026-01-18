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
def parse_bandstructure(self, bandstructure_filename=None) -> None:
    """
        Parse a CP2K bandstructure file.

        Args:
            bandstructure_filename: Filename containing bandstructure info. If
            not provided, then the pmg name of "BAND.bs" will be assumed by
            the filename parser.
        """
    if not bandstructure_filename:
        if self.filenames['band_structure']:
            bandstructure_filename = self.filenames['band_structure'][0]
        else:
            return
    with open(bandstructure_filename, encoding='utf-8') as file:
        lines = file.read().split('\n')
    data = np.loadtxt(bandstructure_filename)
    nkpts = int(lines[0].split()[6])
    nbands = int(lines[0].split()[-2])
    rec_lat = self.final_structure.lattice.reciprocal_lattice if self.final_structure else self.initial_structure.lattice.reciprocal_lattice
    labels = {}
    kpts = []
    nkpts = 0
    for line in lines:
        if not line.startswith('#'):
            continue
        if line.split()[1] == 'Set':
            nkpts += int(lines[0].split()[6])
        elif line.split()[1] == 'Point':
            kpts.append(list(map(float, line.split()[-4:-1])))
        elif line.split()[1] == 'Special':
            splt = line.split()
            label = splt[7]
            if label.upper() == 'GAMMA':
                label = '\\Gamma'
            kpt = np.array(splt[4:7]).astype(float).tolist()
            if label.upper() != 'NONE':
                labels[label] = kpt
    if self.spin_polarized:
        kpts = kpts[::2]
    eigenvals = {}
    if self.spin_polarized:
        up = data.reshape(-1, nbands * 2, data.shape[1])[:, :nbands].reshape(-1, data.shape[1])
        down = data.reshape(-1, nbands * 2, data.shape[1])[:, nbands:].reshape(-1, data.shape[1])
        eigenvals = {Spin.up: up[:, 1].reshape((nkpts, nbands)).T.tolist(), Spin.down: down[:, 1].reshape((nkpts, nbands)).T.tolist()}
    else:
        eigenvals = {Spin.up: data.reshape((nbands, nkpts))}
    occ = data[:, 1][data[:, -1] != 0.0]
    homo = np.max(occ)
    unocc = data[:, 1][data[:, -1] == 0.0]
    lumo = np.min(unocc)
    efermi = (lumo + homo) / 2
    self.efermi = efermi
    self.data['band_structure'] = BandStructureSymmLine(kpoints=kpts, eigenvals=eigenvals, lattice=rec_lat, efermi=efermi, labels_dict=labels, structure=self.final_structure, projections=None)
    self.band_gap = self.data['band_structure'].get_band_gap().get('energy')
    self.vbm = self.data['band_structure'].get_vbm().get('energy')
    self.cbm = self.data['band_structure'].get_cbm().get('energy')