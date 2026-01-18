import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
@property
def simulated_state_disturbance(self):
    """
        Random draw of the state disturbanc e vector from its conditional
        distribution.

        Notes
        -----

        .. math::

            \\eta ~ N(\\hat \\eta, Var(\\hat \\eta \\mid Y_n))
        """
    if self._simulated_state_disturbance is None:
        self._simulated_state_disturbance = np.array(self._simulation_smoother.simulated_state_disturbance, copy=True)
    return self._simulated_state_disturbance