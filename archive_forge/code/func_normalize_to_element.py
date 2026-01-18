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
def normalize_to_element(self, element: Species | Element, factor: float=1) -> None:
    """
        Normalizes the reaction to one of the elements.
        By default, normalizes such that the amount of the element is 1.
        Another factor can be specified.

        Args:
            element (Element/Species): Element to normalize to.
            factor (float): Factor to normalize to. Defaults to 1.
        """
    all_comp = self._all_comp
    coeffs = self._coeffs
    current_el_amount = sum((all_comp[i][element] * abs(coeffs[i]) for i in range(len(all_comp)))) / 2
    scale_factor = factor / current_el_amount
    self._coeffs = [c * scale_factor for c in coeffs]