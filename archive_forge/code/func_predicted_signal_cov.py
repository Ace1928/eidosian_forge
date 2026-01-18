import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
@property
def predicted_signal_cov(self):
    if self._predicted_signal_cov is None:
        self._predicted_signal, self._predicted_signal_cov = self._compute_forecasts(self.predicted_state, self.predicted_state_cov, signal_only=True)
    return self._predicted_signal_cov