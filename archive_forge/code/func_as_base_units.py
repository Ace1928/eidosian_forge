from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
@property
def as_base_units(self):
    """Returns this ArrayWithUnit in base SI units, including derived units.

        Returns:
            An ArrayWithUnit object in base SI units
        """
    return self.to(self.unit.as_base_units[0])