import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
def has_samecoef(self, other):
    """Check if coefficients match.

        .. versionadded:: 1.6.0

        Parameters
        ----------
        other : class instance
            The other class must have the ``coef`` attribute.

        Returns
        -------
        bool : boolean
            True if the coefficients are the same, False otherwise.

        """
    if len(self.coef) != len(other.coef):
        return False
    elif not np.all(self.coef == other.coef):
        return False
    else:
        return True