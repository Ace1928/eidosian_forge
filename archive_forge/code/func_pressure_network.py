import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
def pressure_network(flow_rates, Qtot, k):
    """Evaluate non-linear equation system representing
    the pressures and flows in a system of n parallel pipes::

        f_i = P_i - P_0, for i = 1..n
        f_0 = sum(Q_i) - Qtot

    where Q_i is the flow rate in pipe i and P_i the pressure in that pipe.
    Pressure is modeled as a P=kQ**2 where k is a valve coefficient and
    Q is the flow rate.

    Parameters
    ----------
    flow_rates : float
        A 1-D array of n flow rates [kg/s].
    k : float
        A 1-D array of n valve coefficients [1/kg m].
    Qtot : float
        A scalar, the total input flow rate [kg/s].

    Returns
    -------
    F : float
        A 1-D array, F[i] == f_i.

    """
    P = k * flow_rates ** 2
    F = np.hstack((P[1:] - P[0], flow_rates.sum() - Qtot))
    return F