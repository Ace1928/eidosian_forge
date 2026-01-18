import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
@property
def smoothed_signal_cov(self):
    if self._smoothed_signal_cov is None:
        self._smoothed_signal, self._smoothed_signal_cov = self._compute_forecasts(self.smoothed_state, self.smoothed_state_cov, signal_only=True)
    return self._smoothed_signal_cov