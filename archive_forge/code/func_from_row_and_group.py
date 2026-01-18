from __future__ import annotations
import ast
import functools
import json
import re
import warnings
from collections import Counter
from enum import Enum, unique
from itertools import combinations, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core.units import SUPPORTED_UNIT_NAMES, FloatWithUnit, Ha_to_eV, Length, Mass, Unit
from pymatgen.util.string import Stringify, formula_double_format
@staticmethod
def from_row_and_group(row: int, group: int) -> Element:
    """Returns an element from a row and group number.
        Important Note: For lanthanoids and actinoids, the row number must
        be 8 and 9, respectively, and the group number must be
        between 3 (La, Ac) and 17 (Lu, Lr). This is different than the
        value for Element(symbol).row and Element(symbol).group for these
        elements.

        Args:
            row (int): (pseudo) row number. This is the
                standard row number except for the lanthanoids
                and actinoids for which it is 8 or 9, respectively.
            group (int): (pseudo) group number. This is the
                standard group number except for the lanthanoids
                and actinoids for which it is 3 (La, Ac) to 17 (Lu, Lr).

        Note:
            The 18 group number system is used, i.e. noble gases are group 18.
        """
    for sym in _pt_data:
        el = Element(sym)
        if 57 <= el.Z <= 71:
            el_pseudorow = 8
            el_pseudogroup = (el.Z - 54) % 32
        elif 89 <= el.Z <= 103:
            el_pseudorow = 9
            el_pseudogroup = (el.Z - 54) % 32
        else:
            el_pseudorow = el.row
            el_pseudogroup = el.group
        if el_pseudorow == row and el_pseudogroup == group:
            return el
    raise ValueError('No element with this row and group!')