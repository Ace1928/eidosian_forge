from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
def obj_with_unit(obj: Any, unit: str) -> FloatWithUnit | ArrayWithUnit | dict[str, FloatWithUnit | ArrayWithUnit]:
    """Returns a FloatWithUnit instance if obj is scalar, a dictionary of
    objects with units if obj is a dict, else an instance of
    ArrayWithFloatWithUnit.

    Args:
        obj (Any): Object to be given a unit.
        unit (str): Specific units (eV, Ha, m, ang, etc.).
    """
    unit_type = _UNAME2UTYPE[unit]
    if isinstance(obj, Number):
        return FloatWithUnit(obj, unit=unit, unit_type=unit_type)
    if isinstance(obj, collections.abc.Mapping):
        return {k: obj_with_unit(v, unit) for k, v in obj.items()}
    return ArrayWithUnit(obj, unit=unit, unit_type=unit_type)