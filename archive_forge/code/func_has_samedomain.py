import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
def has_samedomain(self, other):
    """Check if domains match.

        .. versionadded:: 1.6.0

        Parameters
        ----------
        other : class instance
            The other class must have the ``domain`` attribute.

        Returns
        -------
        bool : boolean
            True if the domains are the same, False otherwise.

        """
    return np.all(self.domain == other.domain)