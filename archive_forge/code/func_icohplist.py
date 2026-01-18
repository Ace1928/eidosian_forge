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
@property
def icohplist(self) -> dict[Any, dict[str, Any]]:
    """Returns: icohplist compatible with older version of this class."""
    icohp_dict = {}
    for key, value in self._icohpcollection._icohplist.items():
        icohp_dict[key] = {'length': value._length, 'number_of_bonds': value._num, 'icohp': value._icohp, 'translation': value._translation, 'orbitals': value._orbitals}
    return icohp_dict