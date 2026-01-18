from __future__ import annotations
import os
import sqlite3
import textwrap
from array import array
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from monty.design_patterns import cached_class
from pymatgen.core.operations import MagSymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.groups import SymmetryGroup, in_array_list
from pymatgen.symmetry.settings import JonesFaithfulTransformation
from pymatgen.util.string import transformation_to_string
@classmethod
def from_og(cls, label: Sequence[int] | str) -> Self:
    """Initialize from Opechowski and Guccione (OG) label or number.

        Args:
            label: OG number supplied as list of 3 ints or OG label as str
        """
    db = sqlite3.connect(MAGSYMM_DATA)
    c = db.cursor()
    if isinstance(label, str):
        c.execute('SELECT BNS_label FROM space_groups WHERE OG_label=?', (label,))
    elif isinstance(label, list):
        c.execute('SELECT BNS_label FROM space_groups WHERE OG1=? and OG2=? and OG3=?', (label[0], label[1], label[2]))
    bns_label = c.fetchone()[0]
    db.close()
    return cls(bns_label)