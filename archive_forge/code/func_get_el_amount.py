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
def get_el_amount(self, element: Element | Species) -> float:
    """
        Returns the amount of the element in the reaction.

        Args:
            element (Element/Species): Element in the reaction

        Returns:
            Amount of that element in the reaction.
        """
    return sum((self._all_comp[i][element] * abs(self._coeffs[i]) for i in range(len(self._all_comp)))) / 2