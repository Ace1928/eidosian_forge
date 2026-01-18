from __future__ import annotations
import logging
import re
from itertools import chain, combinations
from typing import TYPE_CHECKING, no_type_check, overload
import numpy as np
from monty.fractions import gcd_float
from monty.json import MontyDecoder, MSONable
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
@property
def normalized_repr(self) -> str:
    """
        A normalized representation of the reaction. All factors are converted
        to lowest common factors.
        """
    return self.normalized_repr_and_factor()[0]