import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
def has_samewindow(self, other):
    """Check if windows match.

        .. versionadded:: 1.6.0

        Parameters
        ----------
        other : class instance
            The other class must have the ``window`` attribute.

        Returns
        -------
        bool : boolean
            True if the windows are the same, False otherwise.

        """
    return np.all(self.window == other.window)