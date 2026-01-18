from __future__ import annotations
import collections
import fnmatch
import os
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import LobsterBandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos, LobsterCompleteDos
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun, VolumetricData
from pymatgen.util.due import Doi, due
def get_volumetricdata_density(self):
    """
        Will return a VolumetricData object including the imaginary part of the wave function.

        Returns:
            VolumetricData
        """
    if not hasattr(self, 'volumetricdata_density'):
        self.set_volumetric_data(self.grid, self.structure)
    return self.volumetricdata_density