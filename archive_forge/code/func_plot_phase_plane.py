from __future__ import (absolute_import, division, print_function)
import numpy as np
from .plotting import plot_result, plot_phase_plane, info_vlines
from .util import import_
def plot_phase_plane(self, indices=None, **kwargs):
    """ Plots a phase portrait from last integration.

        Parameters
        ----------
        indices : iterable of int
        names : iterable of str
        \\*\\*kwargs:
            See :func:`pyodesys.plotting.plot_phase_plane`

        """
    return self._plot(plot_phase_plane, indices=indices, **kwargs)