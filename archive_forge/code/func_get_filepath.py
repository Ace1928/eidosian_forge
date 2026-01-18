from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
def get_filepath(filename, warning, path, suffix):
    """
    Args:
        filename: Filename
        warning: Warning message
        path: Path to search
        suffix: Suffixes to search.
    """
    paths = glob(os.path.join(path, filename + suffix + '*'))
    if not paths:
        warnings.warn(warning)
        return None
    if len(paths) > 1:
        paths.sort(reverse=True)
        warnings.warn(f'Multiple files detected, using {os.path.basename(path)}')
    return paths[0]