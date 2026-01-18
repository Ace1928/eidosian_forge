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
def has_good_quality_maxDeviation(self, limit_maxDeviation: float=0.1) -> bool:
    """
        Will check if the maxDeviation from the ideal bandoverlap is smaller or equal to limit_maxDeviation

        Args:
            limit_maxDeviation: limit of the maxDeviation

        Returns:
            Boolean that will give you information about the quality of the projection.
        """
    return all((deviation <= limit_maxDeviation for deviation in self.max_deviation))