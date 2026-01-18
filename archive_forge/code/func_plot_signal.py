from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def plot_signal(self, signal, **kwargs):
    """Plot a signal on that graph.

        See :func:`pygsp.plotting.plot_signal`.
        """
    from pygsp import plotting
    plotting.plot_signal(self, signal, **kwargs)