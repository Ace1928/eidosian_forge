from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
def get_conversion_factor(self, new_unit):
    """Returns a conversion factor between this unit and a new unit.
        Compound units are supported, but must have the same powers in each
        unit type.

        Args:
            new_unit: The new unit.
        """
    old_base, old_factor = self.as_base_units
    new_base, new_factor = Unit(new_unit).as_base_units
    units_new = sorted(new_base.items(), key=lambda d: _UNAME2UTYPE[d[0]])
    units_old = sorted(old_base.items(), key=lambda d: _UNAME2UTYPE[d[0]])
    factor = old_factor / new_factor
    for old, new in zip(units_old, units_new):
        if old[1] != new[1]:
            raise UnitError(f'Units {old} and {new} are not compatible!')
        c = ALL_UNITS[_UNAME2UTYPE[old[0]]]
        factor *= (c[old[0]] / c[new[0]]) ** old[1]
    return factor