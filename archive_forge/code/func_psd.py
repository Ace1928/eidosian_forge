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
@_preprocess_data(replace_names=['x'])
@_docstring.dedent_interpd
def psd(self, x, NFFT=None, Fs=None, Fc=None, detrend=None, window=None, noverlap=None, pad_to=None, sides=None, scale_by_freq=None, return_line=None, **kwargs):
    """
        Plot the power spectral density.

        The power spectral density :math:`P_{xx}` by Welch's average
        periodogram method.  The vector *x* is divided into *NFFT* length
        segments.  Each segment is detrended by function *detrend* and
        windowed by function *window*.  *noverlap* gives the length of
        the overlap between segments.  The :math:`|\\mathrm{fft}(i)|^2`
        of each segment :math:`i` are averaged to compute :math:`P_{xx}`,
        with a scaling to correct for power loss due to windowing.

        If len(*x*) < *NFFT*, it will be zero padded to *NFFT*.

        Parameters
        ----------
        x : 1-D array or sequence
            Array or sequence containing the data

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
        Pxx : 1-D array
            The values for the power spectrum :math:`P_{xx}` before scaling
            (real valued).

        freqs : 1-D array
            The frequencies corresponding to the elements in *Pxx*.

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
        specgram
            Differs in the default overlap; in not returning the mean of the
            segment periodograms; in returning the times of the segments; and
            in plotting a colormap instead of a line.
        magnitude_spectrum
            Plots the magnitude spectrum.
        csd
            Plots the spectral density between two signals.

        Notes
        -----
        For plotting, the power is plotted as
        :math:`10\\log_{10}(P_{xx})` for decibels, though *Pxx* itself
        is returned.

        References
        ----------
        Bendat & Piersol -- Random Data: Analysis and Measurement Procedures,
        John Wiley & Sons (1986)
        """
    if Fc is None:
        Fc = 0
    pxx, freqs = mlab.psd(x=x, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    freqs += Fc
    if scale_by_freq in (None, True):
        psd_units = 'dB/Hz'
    else:
        psd_units = 'dB'
    line = self.plot(freqs, 10 * np.log10(pxx), **kwargs)
    self.set_xlabel('Frequency')
    self.set_ylabel('Power Spectral Density (%s)' % psd_units)
    self.grid(True)
    vmin, vmax = self.get_ybound()
    step = max(10 * int(np.log10(vmax - vmin)), 1)
    ticks = np.arange(math.floor(vmin), math.ceil(vmax) + 1, step)
    self.set_yticks(ticks)
    if return_line is None or not return_line:
        return (pxx, freqs)
    else:
        return (pxx, freqs, line)