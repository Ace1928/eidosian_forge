import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def get_return_code(self):
    """Extracts the return code for the integration to enable better control
        if the integration fails.

        In general, a return code > 0 implies success, while a return code < 0
        implies failure.

        Notes
        -----
        This section describes possible return codes and their meaning, for available
        integrators that can be selected by `set_integrator` method.

        "vode"

        ===========  =======
        Return Code  Message
        ===========  =======
        2            Integration successful.
        -1           Excess work done on this call. (Perhaps wrong MF.)
        -2           Excess accuracy requested. (Tolerances too small.)
        -3           Illegal input detected. (See printed message.)
        -4           Repeated error test failures. (Check all input.)
        -5           Repeated convergence failures. (Perhaps bad Jacobian
                     supplied or wrong choice of MF or tolerances.)
        -6           Error weight became zero during problem. (Solution
                     component i vanished, and ATOL or ATOL(i) = 0.)
        ===========  =======

        "zvode"

        ===========  =======
        Return Code  Message
        ===========  =======
        2            Integration successful.
        -1           Excess work done on this call. (Perhaps wrong MF.)
        -2           Excess accuracy requested. (Tolerances too small.)
        -3           Illegal input detected. (See printed message.)
        -4           Repeated error test failures. (Check all input.)
        -5           Repeated convergence failures. (Perhaps bad Jacobian
                     supplied or wrong choice of MF or tolerances.)
        -6           Error weight became zero during problem. (Solution
                     component i vanished, and ATOL or ATOL(i) = 0.)
        ===========  =======

        "dopri5"

        ===========  =======
        Return Code  Message
        ===========  =======
        1            Integration successful.
        2            Integration successful (interrupted by solout).
        -1           Input is not consistent.
        -2           Larger nsteps is needed.
        -3           Step size becomes too small.
        -4           Problem is probably stiff (interrupted).
        ===========  =======

        "dop853"

        ===========  =======
        Return Code  Message
        ===========  =======
        1            Integration successful.
        2            Integration successful (interrupted by solout).
        -1           Input is not consistent.
        -2           Larger nsteps is needed.
        -3           Step size becomes too small.
        -4           Problem is probably stiff (interrupted).
        ===========  =======

        "lsoda"

        ===========  =======
        Return Code  Message
        ===========  =======
        2            Integration successful.
        -1           Excess work done on this call (perhaps wrong Dfun type).
        -2           Excess accuracy requested (tolerances too small).
        -3           Illegal input detected (internal error).
        -4           Repeated error test failures (internal error).
        -5           Repeated convergence failures (perhaps bad Jacobian or tolerances).
        -6           Error weight became zero during problem.
        -7           Internal workspace insufficient to finish (internal error).
        ===========  =======
        """
    try:
        self._integrator
    except AttributeError:
        self.set_integrator('')
    return self._integrator.istate