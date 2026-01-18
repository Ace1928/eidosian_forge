import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
def set_stability_method(self, stability_method=None, **kwargs):
    """
        Set the numerical stability method

        The Kalman filter is a recursive algorithm that may in some cases
        suffer issues with numerical stability. The stability method controls
        what, if any, measures are taken to promote stability.

        Parameters
        ----------
        stability_method : int, optional
            Bitmask value to set the stability method to. See notes for
            details.
        **kwargs
            Keyword arguments may be used to influence the stability method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The stability method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        STABILITY_FORCE_SYMMETRY = 0x01
            If this flag is set, symmetry of the predicted state covariance
            matrix is enforced at each iteration of the filter, where each
            element is set to the average of the corresponding elements in the
            upper and lower triangle.

        If the bitmask is set directly via the `stability_method` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the stability method may also be specified by directly
        modifying the class attributes which are defined similarly to the
        keyword arguments.

        The default stability method is `STABILITY_FORCE_SYMMETRY`

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.ssm.stability_method
        1
        >>> mod.ssm.stability_force_symmetry
        True
        >>> mod.ssm.stability_force_symmetry = False
        >>> mod.ssm.stability_method
        0
        """
    if stability_method is not None:
        self.stability_method = stability_method
    for name in KalmanFilter.stability_methods:
        if name in kwargs:
            setattr(self, name, kwargs[name])