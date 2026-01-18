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
def estimate_nbands(self) -> int:
    """
        Estimate the number of bands that VASP will initialize a
        calculation with by default. Note that in practice this
        can depend on # of cores (if not set explicitly).
        Note that this formula is slightly different than the formula on the VASP wiki
        (as of July 2023). This is because the formula in the source code (`main.F`) is
        slightly different than what is on the wiki.
        """
    if self.structure is None:
        raise RuntimeError('No structure is associated with the input set!')
    n_ions = len(self.structure)
    if self.incar['ISPIN'] == 1:
        n_mag = 0
    else:
        n_mag = sum(self.incar['MAGMOM'])
        n_mag = np.floor((n_mag + 1) / 2)
    possible_val_1 = np.floor((self.nelect + 2) / 2) + max(np.floor(n_ions / 2), 3)
    possible_val_2 = np.floor(self.nelect * 0.6)
    n_bands = max(possible_val_1, possible_val_2) + n_mag
    if self.incar.get('LNONCOLLINEAR') is True:
        n_bands = n_bands * 2
    if (n_par := self.incar.get('NPAR')):
        n_bands = np.floor((n_bands + n_par - 1) / n_par) * n_par
    return int(n_bands)