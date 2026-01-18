from __future__ import annotations
import logging
import os.path
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.dev import requires
from monty.functools import lazy_property
from monty.string import marquee
from pymatgen.core.structure import Structure
from pymatgen.core.units import ArrayWithUnit
from pymatgen.core.xcfunc import XcFunc
def read_dimvalue(self, dimname, path='/', default=NO_DEFAULT):
    """
        Returns the value of a dimension.

        Args:
            dimname: Name of the variable
            path: path to the group.
            default: return `default` if `dimname` is not present and
                `default` is not `NO_DEFAULT` else raise self.Error.
        """
    try:
        dim = self._read_dimensions(dimname, path=path)[0]
        return len(dim)
    except self.Error:
        if default is NO_DEFAULT:
            raise
        return default