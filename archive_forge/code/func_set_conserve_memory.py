import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
def set_conserve_memory(self, conserve_memory=None, **kwargs):
    """
        Set the memory conservation method

        By default, the Kalman filter computes a number of intermediate
        matrices at each iteration. The memory conservation options control
        which of those matrices are stored.

        Parameters
        ----------
        conserve_memory : int, optional
            Bitmask value to set the memory conservation method to. See notes
            for details.
        **kwargs
            Keyword arguments may be used to influence the memory conservation
            method by setting individual boolean flags. See notes for details.

        Notes
        -----
        The memory conservation method is defined by a collection of boolean
        flags, and is internally stored as a bitmask. The methods available
        are:

        MEMORY_STORE_ALL
            Store all intermediate matrices. This is the default value.
        MEMORY_NO_FORECAST_MEAN
            Do not store the forecast or forecast errors. If this option is
            used, the `predict` method from the results class is unavailable.
        MEMORY_NO_FORECAST_COV
            Do not store the forecast error covariance matrices.
        MEMORY_NO_FORECAST
            Do not store the forecast, forecast error, or forecast error
            covariance matrices. If this option is used, the `predict` method
            from the results class is unavailable.
        MEMORY_NO_PREDICTED_MEAN
            Do not store the predicted state.
        MEMORY_NO_PREDICTED_COV
            Do not store the predicted state covariance
            matrices.
        MEMORY_NO_PREDICTED
            Do not store the predicted state or predicted state covariance
            matrices.
        MEMORY_NO_FILTERED_MEAN
            Do not store the filtered state.
        MEMORY_NO_FILTERED_COV
            Do not store the filtered state covariance
            matrices.
        MEMORY_NO_FILTERED
            Do not store the filtered state or filtered state covariance
            matrices.
        MEMORY_NO_LIKELIHOOD
            Do not store the vector of loglikelihood values for each
            observation. Only the sum of the loglikelihood values is stored.
        MEMORY_NO_GAIN
            Do not store the Kalman gain matrices.
        MEMORY_NO_SMOOTHING
            Do not store temporary variables related to Kalman smoothing. If
            this option is used, smoothing is unavailable.
        MEMORY_NO_STD_FORECAST
            Do not store standardized forecast errors.
        MEMORY_CONSERVE
            Do not store any intermediate matrices.

        If the bitmask is set directly via the `conserve_memory` argument,
        then the full method must be provided.

        If keyword arguments are used to set individual boolean flags, then
        the lowercase of the method must be used as an argument name, and the
        value is the desired value of the boolean flag (True or False).

        Note that the memory conservation method may also be specified by
        directly modifying the class attributes which are defined similarly to
        the keyword arguments.

        The default memory conservation method is `MEMORY_STORE_ALL`, so that
        all intermediate matrices are stored.

        Examples
        --------
        >>> mod = sm.tsa.statespace.SARIMAX(range(10))
        >>> mod.ssm..conserve_memory
        0
        >>> mod.ssm.memory_no_predicted
        False
        >>> mod.ssm.memory_no_predicted = True
        >>> mod.ssm.conserve_memory
        2
        >>> mod.ssm.set_conserve_memory(memory_no_filtered=True,
        ...                             memory_no_forecast=True)
        >>> mod.ssm.conserve_memory
        7
        """
    if conserve_memory is not None:
        self.conserve_memory = conserve_memory
    for name in KalmanFilter.memory_options:
        if name in kwargs:
            setattr(self, name, kwargs[name])