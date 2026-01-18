from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from pymatgen.analysis.chemenv.utils.math_utils import (
def fractions(self, data):
    """Get the fractions from the CSM ratio function applied to the data.

        Args:
            data: List of CSM values to estimate fractions.

        Returns:
            Corresponding fractions for each CSM.
        """
    if len(data) == 0:
        return None
    close_to_zero = np.isclose(data, 0.0, atol=1e-10).tolist()
    n_zeros = close_to_zero.count(True)
    if n_zeros == 1:
        fractions = [0.0] * len(data)
        fractions[close_to_zero.index(True)] = 1.0
        return fractions
    if n_zeros > 1:
        raise RuntimeError('Should not have more than one continuous symmetry measure with value equal to 0.0')
    fractions = self.eval(np.array(data))
    total = np.sum(fractions)
    if total > 0.0:
        return fractions / total
    return None