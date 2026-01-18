from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
def conversions(self):
    """Returns a string showing the available conversions.
        Useful tool in interactive mode.
        """
    return '\n'.join((str(self.to(unit)) for unit in self.supported_units))