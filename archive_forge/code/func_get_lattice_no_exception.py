from __future__ import annotations
import math
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from functools import partial
from inspect import getfullargspec
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.core import Composition, DummySpecies, Element, Lattice, PeriodicSite, Species, Structure, get_el_sp
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
@staticmethod
def get_lattice_no_exception(data, length_strings=('a', 'b', 'c'), angle_strings=('alpha', 'beta', 'gamma'), lattice_type=None):
    """
        Take a dictionary of CIF data and returns a pymatgen Lattice object.

        Args:
            data: a dictionary of the CIF file
            length_strings: The strings that are used to identify the length parameters in the CIF file.
            angle_strings: The strings that are used to identify the angles in the CIF file.
            lattice_type: The type of lattice.  This is a string, and can be any of the following:

        Returns:
            Lattice object
        """
    lengths = [str2float(data['_cell_length_' + i]) for i in length_strings]
    angles = [str2float(data['_cell_angle_' + i]) for i in angle_strings]
    if not lattice_type:
        return Lattice.from_parameters(*lengths, *angles)
    return getattr(Lattice, lattice_type)(*lengths + angles)