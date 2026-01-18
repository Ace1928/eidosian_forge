from __future__ import annotations
import re
from collections import defaultdict
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.feff import Header, Potential, Tags
@property
def material_formula(self):
    """Returns chemical formula of material from feff.inp file."""
    try:
        form = self.header.formula
    except IndexError:
        form = 'No formula provided'
    return ''.join(map(str, form))