from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from pymatgen.analysis.chemenv.utils.math_utils import (
def smoothstep(self, vals):
    """Get the evaluation of the smoothstep ratio function: f(x)=3*x^2-2*x^3.

        The CSM values (i.e. "x"), are scaled between the "lower_csm" and "upper_csm" parameters.

        Args:
            vals: CSM values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the CSM values.
        """
    return smootherstep(vals, edges=[self.__dict__['lower_csm'], self.__dict__['upper_csm']], inverse=True)