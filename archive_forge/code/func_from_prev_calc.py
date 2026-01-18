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
@classmethod
def from_prev_calc(cls, prev_calc_dir: str, mode: str='DIAG', **kwargs) -> Self:
    """
        Generate a set of VASP input files for GW or BSE calculations from a
        directory of previous Exact Diag VASP run.

        Args:
            prev_calc_dir (str): The directory contains the outputs(
                vasprun.xml of previous vasp run.
            mode (str): Supported modes are "STATIC", "DIAG" (default), "GW",
                and "BSE".
            **kwargs: All kwargs supported by MVLGWSet, other than structure,
                prev_incar and mode, which are determined from the
                prev_calc_dir.
        """
    input_set = cls(_dummy_structure, mode=mode, **kwargs)
    return input_set.override_from_prev_calc(prev_calc_dir=prev_calc_dir)