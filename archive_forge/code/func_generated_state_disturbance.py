import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
@property
def generated_state_disturbance(self):
    """
        Randomly drawn state disturbance variates, used to construct
        `generated_state` and `generated_obs`.

        Notes
        -----

        .. math::

            \\eta_t^+ ~ N(0, Q_t)

        If `disturbance_variates` were provided to the `simulate()` method,
        then this returns those variates (which were N(0,1)) transformed to the
        distribution above.
        """
    if self._generated_state_disturbance is None:
        self._generated_state_disturbance = np.array(self._simulation_smoother.state_disturbance_variates, copy=True).reshape(self.model.nobs, self.model.k_posdef)
    return self._generated_state_disturbance