import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
def set_filter_timing(self, alternate_timing=None, **kwargs):
    """
        Set the filter timing convention

        By default, the Kalman filter follows Durbin and Koopman, 2012, in
        initializing the filter with predicted values. Kim and Nelson, 1999,
        instead initialize the filter with filtered values, which is
        essentially just a different timing convention.

        Parameters
        ----------
        alternate_timing : int, optional
            Whether or not to use the alternate timing convention. Default is
            unspecified.
        **kwargs
            Keyword arguments may be used to influence the memory conservation
            method by setting individual boolean flags. See notes for details.
        """
    if alternate_timing is not None:
        self.filter_timing = int(alternate_timing)
    if 'timing_init_predicted' in kwargs:
        self.filter_timing = int(not kwargs['timing_init_predicted'])
    if 'timing_init_filtered' in kwargs:
        self.filter_timing = int(kwargs['timing_init_filtered'])