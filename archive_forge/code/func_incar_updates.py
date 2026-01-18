from __future__ import annotations
import abc
import itertools
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast
from zipfile import ZipFile
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, SiteCollection, Species, Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
@property
def incar_updates(self) -> dict:
    """Get updates to the INCAR config for this calculation type."""
    updates = {'ALGO': 'Exact', 'EDIFF': 1e-08, 'IBRION': -1, 'ICHARG': 1, 'ISMEAR': 0, 'SIGMA': 0.01, 'LWAVE': True, 'LREAL': False, 'NELM': 100, 'NSW': 0, 'LOPTICS': True, 'CSHIFT': 0.1, 'NEDOS': self.nedos}
    if self.mode == 'RPA':
        updates.update({'ALGO': 'CHI', 'NELM': 1, 'NOMEGA': 1000, 'EDIFF': None, 'LOPTICS': None, 'LWAVE': None})
    if self.prev_vasprun is not None and self.mode == 'IPA':
        prev_nbands = int(self.prev_vasprun.parameters['NBANDS']) if self.nbands is None else self.nbands
        updates['NBANDS'] = int(np.ceil(prev_nbands * self.nbands_factor))
    if self.prev_vasprun is not None and self.mode == 'RPA':
        self.nkred = self.prev_vasprun.kpoints.kpts[0] if self.nkred is None else self.nkred
        updates.update({'NKREDX': self.nkred[0], 'NKREDY': self.nkred[1], 'NKREDZ': self.nkred[2]})
    return updates