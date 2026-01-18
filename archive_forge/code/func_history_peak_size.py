import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def history_peak_size(self, include_inputs=True):
    """Get the peak combined intermediate size of this computation.

        Parameters
        ----------
        include_inputs : bool, optional
            Whether to include the size of the inputs in the computation. If
            ``True`` It is assumed they can be garbage collected once used but
            are all present at the beginning of the computation.
        """
    return max(self.history_size_footprint(include_inputs=include_inputs))