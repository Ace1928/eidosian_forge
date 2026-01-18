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
def parse_magmoms(data, lattice=None):
    """Parse atomic magnetic moments from data dictionary."""
    if lattice is None:
        raise Exception('Magmoms given in terms of crystal axes in magCIF spec.')
    try:
        magmoms = {data['_atom_site_moment_label'][i]: np.array([str2float(data['_atom_site_moment_crystalaxis_x'][i]), str2float(data['_atom_site_moment_crystalaxis_y'][i]), str2float(data['_atom_site_moment_crystalaxis_z'][i])]) for i in range(len(data['_atom_site_moment_label']))}
    except (ValueError, KeyError):
        return None
    return magmoms