import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
def set_job(self, fit_type=None, deriv=None, var_calc=None, del_init=None, restart=None):
    """
        Sets the "job" parameter is a hopefully comprehensible way.

        If an argument is not specified, then the value is left as is. The
        default value from class initialization is for all of these options set
        to 0.

        Parameters
        ----------
        fit_type : {0, 1, 2} int
            0 -> explicit ODR

            1 -> implicit ODR

            2 -> ordinary least-squares
        deriv : {0, 1, 2, 3} int
            0 -> forward finite differences

            1 -> central finite differences

            2 -> user-supplied derivatives (Jacobians) with results
              checked by ODRPACK

            3 -> user-supplied derivatives, no checking
        var_calc : {0, 1, 2} int
            0 -> calculate asymptotic covariance matrix and fit
                 parameter uncertainties (V_B, s_B) using derivatives
                 recomputed at the final solution

            1 -> calculate V_B and s_B using derivatives from last iteration

            2 -> do not calculate V_B and s_B
        del_init : {0, 1} int
            0 -> initial input variable offsets set to 0

            1 -> initial offsets provided by user in variable "work"
        restart : {0, 1} int
            0 -> fit is not a restart

            1 -> fit is a restart

        Notes
        -----
        The permissible values are different from those given on pg. 31 of the
        ODRPACK User's Guide only in that one cannot specify numbers greater than
        the last value for each variable.

        If one does not supply functions to compute the Jacobians, the fitting
        procedure will change deriv to 0, finite differences, as a default. To
        initialize the input variable offsets by yourself, set del_init to 1 and
        put the offsets into the "work" variable correctly.

        """
    if self.job is None:
        job_l = [0, 0, 0, 0, 0]
    else:
        job_l = [self.job // 10000 % 10, self.job // 1000 % 10, self.job // 100 % 10, self.job // 10 % 10, self.job % 10]
    if fit_type in (0, 1, 2):
        job_l[4] = fit_type
    if deriv in (0, 1, 2, 3):
        job_l[3] = deriv
    if var_calc in (0, 1, 2):
        job_l[2] = var_calc
    if del_init in (0, 1):
        job_l[1] = del_init
    if restart in (0, 1):
        job_l[0] = restart
    self.job = job_l[0] * 10000 + job_l[1] * 1000 + job_l[2] * 100 + job_l[3] * 10 + job_l[4]