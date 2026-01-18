from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def plot_spectrogram(self, **kwargs):
    """Plot the graph's spectrogram.

        See :func:`pygsp.plotting.plot_spectrogram`.
        """
    from pygsp import plotting
    plotting.plot_spectrogram(self, **kwargs)