from __future__ import annotations
import abc
import copy
import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Union
import numpy as np
from monty.design_patterns import cached_class
from monty.json import MSONable
from monty.serialization import loadfn
from tqdm import tqdm
from uncertainties import ufloat
from pymatgen.analysis.structure_analyzer import oxide_type, sulfide_type
from pymatgen.core import SETTINGS, Composition, Element
from pymatgen.entries.computed_entries import (
from pymatgen.io.vasp.sets import MITRelaxSet, MPRelaxSet, VaspInputSet
from pymatgen.util.due import Doi, due
def get_correction(self, entry) -> ufloat:
    """
        Args:
            entry: A ComputedEntry/ComputedStructureEntry.

        Returns:
            Correction, Uncertainty.
        """
    calc_u = entry.parameters.get('hubbards') or defaultdict(int)
    comp = entry.composition
    elements = sorted((el for el in comp.elements if comp[el] > 0), key=lambda el: el.X)
    most_electro_neg = elements[-1].symbol
    correction = ufloat(0.0, 0.0)
    u_corr = self.u_corrections.get(most_electro_neg, {})
    u_settings = self.u_settings.get(most_electro_neg, {})
    u_errors = self.u_errors.get(most_electro_neg, defaultdict(float))
    for el in comp.elements:
        sym = el.symbol
        if calc_u.get(sym, 0) != u_settings.get(sym, 0):
            raise CompatibilityError(f'Invalid U value of {calc_u.get(sym, 0)} on {sym}')
        if sym in u_corr:
            correction += ufloat(u_corr[sym], u_errors[sym]) * comp[el]
    return correction