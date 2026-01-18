import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
def cutdeg(self, deg):
    """Truncate series to the given degree.

        Reduce the degree of the series to `deg` by discarding the
        high order terms. If `deg` is greater than the current degree a
        copy of the current series is returned. This can be useful in least
        squares where the coefficients of the high degree terms may be very
        small.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        deg : non-negative int
            The series is reduced to degree `deg` by discarding the high
            order terms. The value of `deg` must be a non-negative integer.

        Returns
        -------
        new_series : series
            New instance of series with reduced degree.

        """
    return self.truncate(deg + 1)