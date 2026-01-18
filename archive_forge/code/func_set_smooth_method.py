import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace import tools, initialization
def set_smooth_method(self, smooth_method=None, **kwargs):
    """
        Set the smoothing method

        The smoothing method can be used to override the Kalman smoother
        approach used. By default, the Kalman smoother used depends on the
        Kalman filter method.

        Parameters
        ----------
        smooth_method : int, optional
            Bitmask value to set the filter method to. See notes for details.
        **kwargs
            Keyword arguments may be used to influence the filter method by
            setting individual boolean flags. See notes for details.

        Notes
        -----
        The smoothing method is defined by a collection of boolean flags, and
        is internally stored as a bitmask. The methods available are:

        SMOOTH_CONVENTIONAL = 0x01
            Default Kalman smoother, as presented in Durbin and Koopman, 2012
            chapter 4.
        SMOOTH_CLASSICAL = 0x02
            Classical Kalman smoother, as presented in Anderson and Moore, 1979
            or Durbin and Koopman, 2012 chapter 4.6.1.
        SMOOTH_ALTERNATIVE = 0x04
            Modified Bryson-Frazier Kalman smoother method; this is identical
            to the conventional method of Durbin and Koopman, 2012, except that
            an additional intermediate step is included.
        SMOOTH_UNIVARIATE = 0x08
            Univariate Kalman smoother, as presented in Durbin and Koopman,
            2012 chapter 6, except with modified Bryson-Frazier timing.

        Practically speaking, these methods should all produce the same output
        but different computational implications, numerical stability
        implications, or internal timing assumptions.

        Note that only the first method is available if using a Scipy version
        older than 0.16.

        If the bitmask is set directly via the `smooth_method` argument, then
        the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the filter method may also be specified by directly modifying
        the class attributes which are defined similarly to the keyword
        arguments.

        The default filtering method is SMOOTH_CONVENTIONAL.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.smooth_method
        1
        >>> mod.filter_conventional
        True
        >>> mod.filter_univariate = True
        >>> mod.smooth_method
        17
        >>> mod.set_smooth_method(filter_univariate=False,
                                  filter_collapsed=True)
        >>> mod.smooth_method
        33
        >>> mod.set_smooth_method(smooth_method=1)
        >>> mod.filter_conventional
        True
        >>> mod.filter_univariate
        False
        >>> mod.filter_collapsed
        False
        >>> mod.filter_univariate = True
        >>> mod.smooth_method
        17
        """
    if smooth_method is not None:
        self.smooth_method = smooth_method
    for name in KalmanSmoother.smooth_methods:
        if name in kwargs:
            setattr(self, name, kwargs[name])