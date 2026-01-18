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
@property
def ellipticity(self):
    """Most meaningful for bond critical points, can be physically interpreted as e.g.
        degree of pi-bonding in organic molecules. Consult literature for more info.

        Returns:
            float: The ellipticity of the field at the critical point.
        """
    eig, _ = np.linalg.eig(self.field_hessian)
    eig.sort()
    return eig[0] / eig[1] - 1