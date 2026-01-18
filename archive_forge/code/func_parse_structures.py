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
def parse_structures(self, trajectory_file=None, lattice_file=None):
    """
        Parses the structures from a cp2k calculation. Static calculations simply use the initial
        structure. For calculations with ionic motion, the function will look for the appropriate
        trajectory and lattice files based on naming convention. If no file is given, and no file
        is found, it is assumed that the lattice/structure remained constant, and the initial
        lattice/structure is used. Cp2k does not output the trajectory in the main output file by
        default, so non static calculations have to reference the trajectory file.
        """
    self.parse_initial_structure()
    trajectory_file = trajectory_file or self.filenames.get('trajectory')
    if isinstance(trajectory_file, list):
        if len(trajectory_file) == 1:
            trajectory_file = trajectory_file[0]
        elif len(trajectory_file) > 1:
            raise FileNotFoundError('Unable to automatically determine trajectory file. More than one exist.')
    if lattice_file is None:
        if len(self.filenames['cell']) == 0:
            lattices = self.parse_cell_params()
        elif len(self.filenames['cell']) == 1:
            latt_file = np.loadtxt(self.filenames['cell'][0])
            lattices = [latt[2:11].reshape(3, 3) for latt in latt_file] if len(latt_file.shape) > 1 else [latt_file[2:11].reshape(3, 3)]
        else:
            raise FileNotFoundError('Unable to automatically determine lattice file. More than one exist.')
    else:
        latt_file = np.loadtxt(lattice_file)
        lattices = [latt[2:].reshape(3, 3) for latt in latt_file]
    if not trajectory_file:
        self.structures = [self.initial_structure]
        self.final_structure = self.structures[-1]
    else:
        mols = XYZ.from_file(trajectory_file).all_molecules
        for mol in mols:
            mol.set_charge_and_spin(charge=self.charge, spin_multiplicity=self.multiplicity)
        self.structures = []
        gs = self.initial_structure.site_properties.get('ghost')
        if not self.is_molecule:
            for mol, latt in zip(mols, lattices):
                self.structures.append(Structure(lattice=latt, coords=[s.coords for s in mol], species=[s.specie for s in mol], coords_are_cartesian=True, site_properties={'ghost': gs} if gs else {}, charge=self.charge))
        else:
            self.structures = mols
        self.final_structure = self.structures[-1]