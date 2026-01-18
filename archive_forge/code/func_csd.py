import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
@_preprocess_data(replace_names=['x', 'y'], label_namer='y')
@_docstring.dedent_interpd
def csd(self, x, y, NFFT=None, Fs=None, Fc=None, detrend=None, window=None, noverlap=None, pad_to=None, sides=None, scale_by_freq=None, return_line=None, **kwargs):
    """
        Plot the cross-spectral density.

        The cross spectral density :math:`P_{xy}` by Welch's average
        periodogram method.  The vectors *x* and *y* are divided into
        *NFFT* length segments.  Each segment is detrended by function
        *detrend* and windowed by function *window*.  *noverlap* gives
        the length of the overlap between segments.  The product of
        the direct FFTs of *x* and *y* are averaged over each segment
        to compute :math:`P_{xy}`, with a scaling to correct for power
        loss due to windowing.

        If len(*x*) < *NFFT* or len(*y*) < *NFFT*, they will be zero
        padded to *NFFT*.

        Parameters
        ----------
        x, y : 1-D arrays or sequences
            Arrays or sequences containing the data.

        %(Spectral)s

        %(PSD)s

        noverlap : int, default: 0 (no overlap)
            The number of points of overlap between segments.

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        return_line : bool, default: False
            Whether to include the line object plotted in the returned values.

        Returns
        -------
        Pxy : 1-D array
            The values for the cross spectrum :math:`P_{xy}` before scaling
            (complex valued).

        freqs : 1-D array
            The frequencies corresponding to the elements in *Pxy*.

        line : `~matplotlib.lines.Line2D`
            The line created by this function.
            Only returned if *return_line* is True.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s

        See Also
        --------
        psd : is equivalent to setting ``y = x``.

        Notes
        -----
        For plotting, the power is plotted as
        :math:`10 \\log_{10}(P_{xy})` for decibels, though :math:`P_{xy}` itself
        is returned.

        References
        ----------
        Bendat & Piersol -- Random Data: Analysis and Measurement Procedures,
        John Wiley & Sons (1986)
        """
    if Fc is None:
        Fc = 0
    pxy, freqs = mlab.csd(x=x, y=y, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    freqs += Fc
    line = self.plot(freqs, 10 * np.log10(np.abs(pxy)), **kwargs)
    self.set_xlabel('Frequency')
    self.set_ylabel('Cross Spectrum Magnitude (dB)')
    self.grid(True)
    vmin, vmax = self.get_ybound()
    step = max(10 * int(np.log10(vmax - vmin)), 1)
    ticks = np.arange(math.floor(vmin), math.ceil(vmax) + 1, step)
    self.set_yticks(ticks)
    if return_line is None or not return_line:
        return (pxy, freqs)
    else:
        return (pxy, freqs, line)