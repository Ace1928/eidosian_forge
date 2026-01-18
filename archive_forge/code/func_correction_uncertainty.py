from __future__ import annotations
import abc
import json
import math
import os
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from scipy.interpolate import interp1d
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.due import Doi, due
@property
def correction_uncertainty(self) -> float:
    """
        Returns:
            float: the uncertainty of the energy adjustments applied to the entry in eV.
        """
    unc = ufloat(0.0, 0.0) + sum((ufloat(ea.value, ea.uncertainty) if not np.isnan(ea.uncertainty) else ufloat(ea.value, 0) for ea in self.energy_adjustments))
    if unc.nominal_value != 0 and unc.std_dev == 0:
        return np.nan
    return unc.std_dev