import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
@contextlib.contextmanager
def fixed_scale(self, scale):
    """
        fixed_scale(scale)

        Context manager for fixing the scale when FILTER_CONCENTRATED is set

        Parameters
        ----------
        scale : numeric
            Scale of the model.

        Notes
        -----
        This a no-op if scale is None.

        This context manager is most useful in models which are explicitly
        concentrating out the scale, so that the set of parameters they are
        estimating does not include the scale.
        """
    if scale is not None and scale != 1:
        if not self.filter_concentrated:
            raise ValueError('Cannot provide scale if filter method does not include FILTER_CONCENTRATED.')
        self.filter_concentrated = False
        self._scale = scale
        obs_cov = self['obs_cov']
        state_cov = self['state_cov']
        self['obs_cov'] = scale * obs_cov
        self['state_cov'] = scale * state_cov
    try:
        yield
    finally:
        if scale is not None and scale != 1:
            self['state_cov'] = state_cov
            self['obs_cov'] = obs_cov
            self.filter_concentrated = True
            self._scale = None