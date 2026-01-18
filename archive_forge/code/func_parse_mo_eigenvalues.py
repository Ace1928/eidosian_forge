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
def parse_mo_eigenvalues(self):
    """
        Parse the MO eigenvalues from the cp2k output file. Will get the eigenvalues (and band gap)
        at each ionic step (if more than one exist).

        Everything is decomposed by spin channel. If calculation was performed without spin
        polarization, then only Spin.up will be present, which represents the average of up and
        down.
        """
    eigenvalues = []
    efermi = []
    with zopen(self.filename, mode='rt') as file:
        lines = iter(file.readlines())
        for line in lines:
            try:
                if ' occupied subspace spin' in line:
                    eigenvalues.append({'occupied': {Spin.up: [], Spin.down: []}, 'unoccupied': {Spin.up: [], Spin.down: []}})
                    efermi.append({Spin.up: np.nan, Spin.down: np.nan})
                    next(lines)
                    while True:
                        line = next(lines)
                        if 'Fermi' in line:
                            efermi[-1][Spin.up] = float(line.split()[-1])
                            break
                        eigenvalues[-1]['occupied'][Spin.up] += [Ha_to_eV * float(val) for val in line.split()]
                    next(lines)
                    line = next(lines)
                    if ' occupied subspace spin' in line:
                        next(lines)
                        while True:
                            line = next(lines)
                            if 'Fermi' in line:
                                efermi[-1][Spin.down] = float(line.split()[-1])
                                break
                            eigenvalues[-1]['occupied'][Spin.down] += [Ha_to_eV * float(val) for val in line.split()]
                if ' unoccupied subspace spin' in line:
                    next(lines)
                    line = next(lines)
                    while True:
                        if 'WARNING : did not converge' in line:
                            warnings.warn('Convergence of eigenvalues for unoccupied subspace spin 1 did NOT converge')
                            next(lines)
                            next(lines)
                            next(lines)
                            line = next(lines)
                            eigenvalues[-1]['unoccupied'][Spin.up] += [Ha_to_eV * float(line) for line in line.split()]
                            next(lines)
                            line = next(lines)
                            break
                        if 'convergence' in line:
                            line = next(lines)
                        if 'eigenvalues' in line.lower() or 'HOMO' in line or '|' in line:
                            break
                        eigenvalues[-1]['unoccupied'][Spin.up] += [Ha_to_eV * float(val) for val in line.split()]
                        line = next(lines)
                    if ' unoccupied subspace spin' in line:
                        next(lines)
                        line = next(lines)
                        while True:
                            if 'WARNING : did not converge' in line:
                                warnings.warn('Convergence of eigenvalues for unoccupied subspace spin 2 did NOT converge')
                                next(lines)
                                next(lines)
                                next(lines)
                                line = next(lines)
                                eigenvalues[-1]['unoccupied'][Spin.down] += [Ha_to_eV * float(line) for line in line.split()]
                                break
                            if 'convergence' in line:
                                line = next(lines)
                            if 'HOMO' in line or '|' in line:
                                next(lines)
                                break
                            try:
                                eigenvalues[-1]['unoccupied'][Spin.down] += [Ha_to_eV * float(val) for val in line.split()]
                            except AttributeError:
                                break
                            line = next(lines)
            except ValueError:
                eigenvalues = [{'occupied': {Spin.up: None, Spin.down: None}, 'unoccupied': {Spin.up: None, Spin.down: None}}]
                warnings.warn('Convergence of eigenvalues for one or more subspaces did NOT converge')
    self.data['eigenvalues'] = eigenvalues
    if len(eigenvalues) == 0:
        return
    if self.spin_polarized:
        self.data['vbm'] = {Spin.up: np.max(eigenvalues[-1]['occupied'][Spin.up]), Spin.down: np.max(eigenvalues[-1]['occupied'][Spin.down])}
        self.data['cbm'] = {Spin.up: np.nanmin(eigenvalues[-1]['unoccupied'][Spin.up] or np.nan), Spin.down: np.nanmin(eigenvalues[-1]['unoccupied'][Spin.down] or np.nan)}
        self.vbm = np.nanmean(list(self.data['vbm'].values()))
        self.cbm = np.nanmean(list(self.data['cbm'].values()))
        self.efermi = np.nanmean(list(efermi[-1].values()))
    else:
        self.data['vbm'] = {Spin.up: np.max(eigenvalues[-1]['occupied'][Spin.up]), Spin.down: None}
        self.data['cbm'] = {Spin.up: np.min(eigenvalues[-1]['unoccupied'][Spin.up]), Spin.down: None}
        self.vbm = self.data['vbm'][Spin.up]
        self.cbm = self.data['cbm'][Spin.up]
        self.efermi = efermi[-1][Spin.up]
    n_occ = len(eigenvalues[-1]['occupied'][Spin.up])
    n_unocc = len(eigenvalues[-1]['unoccupied'][Spin.up])
    self.data['tdos'] = Dos(efermi=self.vbm + 1e-06, energies=list(eigenvalues[-1]['occupied'][Spin.up]) + list(eigenvalues[-1]['unoccupied'][Spin.down]), densities={Spin.up: [1 for _ in range(n_occ)] + [0 for _ in range(n_unocc)], Spin.down: [1 for _ in range(n_occ)] + [0 for _ in range(n_unocc)]})