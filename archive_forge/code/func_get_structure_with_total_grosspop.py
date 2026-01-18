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
def get_structure_with_total_grosspop(self, structure_filename: str) -> Structure:
    """
        Get a Structure with Mulliken and Loewdin total grosspopulations as site properties

        Args:
            structure_filename (str): filename of POSCAR

        Returns:
            Structure Object with Mulliken and Loewdin total grosspopulations as site properties.
        """
    struct = Structure.from_file(structure_filename)
    mullikengp = []
    loewdingp = []
    for grosspop in self.list_dict_grosspop:
        mullikengp += [grosspop['Mulliken GP']['total']]
        loewdingp += [grosspop['Loewdin GP']['total']]
    site_properties = {'Total Mulliken GP': mullikengp, 'Total Loewdin GP': loewdingp}
    return struct.copy(site_properties=site_properties)