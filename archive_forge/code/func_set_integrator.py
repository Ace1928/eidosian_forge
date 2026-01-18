import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def set_integrator(self, name, **integrator_params):
    """
        Set integrator by name.

        Parameters
        ----------
        name : str
            Name of the integrator
        **integrator_params
            Additional parameters for the integrator.
        """
    if name == 'zvode':
        raise ValueError('zvode must be used with ode, not complex_ode')
    lband = integrator_params.get('lband')
    uband = integrator_params.get('uband')
    if lband is not None or uband is not None:
        integrator_params['lband'] = 2 * (lband or 0) + 1
        integrator_params['uband'] = 2 * (uband or 0) + 1
    return ode.set_integrator(self, name, **integrator_params)