import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
@property
def kalman_gain(self):
    """
        Kalman gain matrices
        """
    if self._kalman_gain is None:
        self._kalman_gain = np.zeros((self.k_states, self.k_endog, self.nobs), dtype=self.dtype)
        for t in range(self.nobs):
            if self.nmissing[t] == self.k_endog:
                continue
            design_t = 0 if self.design.shape[2] == 1 else t
            transition_t = 0 if self.transition.shape[2] == 1 else t
            if self.nmissing[t] == 0:
                self._kalman_gain[:, :, t] = np.dot(np.dot(self.transition[:, :, transition_t], self.predicted_state_cov[:, :, t]), np.dot(np.transpose(self.design[:, :, design_t]), np.linalg.inv(self.forecasts_error_cov[:, :, t])))
            else:
                mask = ~self.missing[:, t].astype(bool)
                F = self.forecasts_error_cov[np.ix_(mask, mask, [t])]
                self._kalman_gain[:, mask, t] = np.dot(np.dot(self.transition[:, :, transition_t], self.predicted_state_cov[:, :, t]), np.dot(np.transpose(self.design[mask, :, design_t]), np.linalg.inv(F[:, :, 0])))
    return self._kalman_gain