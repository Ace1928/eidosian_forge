from __future__ import (absolute_import, division, print_function)
from math import log
import numpy as np
def plot_result(x, y, indices=None, plot_kwargs_cb=None, ax=None, ls=('-', '--', ':', '-.'), c=('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black'), m=('o', 'v', '8', 's', 'p', 'x', '+', 'd', 's'), m_lim=-1, lines=None, interpolate=None, interp_from_deriv=None, names=None, latex_names=None, xlabel=None, ylabel=None, xscale=None, yscale=None, legend=False, yerr=None, labels=None, tex_lbl_fmt='$%s$', fig_kw=None, xlim=None, ylim=None):
    """
    Plot the depepndent variables vs. the independent variable

    Parameters
    ----------
    x : array_like
        Values of the independent variable.
    y : array_like
        Values of the independent variable. This must hold
        ``y.shape[0] == len(x)``, plot_results will draw
        ``y.shape[1]`` lines. If ``interpolate != None``
        y is expected two be three dimensional, otherwise two dimensional.
    indices : iterable of integers
        What indices to plot (default: None => all).
    plot : callback (default: None)
        If None, use ``matplotlib.pyplot.plot``.
    plot_kwargs_cb : callback(int) -> dict
        Keyword arguments for plot for each index (0:len(y)-1).
    ax : Axes
    ls : iterable
        Linestyles to cycle through (only used if plot and plot_kwargs_cb
        are both None).
    c : iterable
        Colors to cycle through (only used if plot and plot_kwargs_cb
        are both None).
    m : iterable
        Markers to cycle through (only used if plot and plot_kwargs_cb
        are both None and m_lim > 0).
    m_lim : int (default: -1)
        Upper limit (exclusive, number of points) for using markers instead of
        lines.
    lines : None
        default: draw between markers unless we are interpolating as well.
    interpolate : bool or int (default: None)
        Density-multiplier for grid of independent variable when interpolating
        if True => 20. negative integer signifies log-spaced grid.
    interp_from_deriv : callback (default: None)
        When ``None``: ``scipy.interpolate.BPoly.from_derivatives``
    names : iterable of str
    latex_names : iterable of str
    labels : iterable of str
        If ``None``, use ``latex_names`` or ``names`` (in that order).

    """
    import matplotlib.pyplot as plt
    if ax is None:
        _fig, ax = plt.subplots(1, 1, **fig_kw or {})
    if plot_kwargs_cb is None:

        def plot_kwargs_cb(idx, lines=False, markers=False, labels=None):
            kw = {'c': c[idx % len(c)]}
            if lines:
                kw['ls'] = ls[idx % len(ls)]
                if isinstance(lines, float):
                    kw['alpha'] = lines
            else:
                kw['ls'] = 'None'
            if markers:
                kw['marker'] = m[idx % len(m)]
            if labels:
                kw['label'] = labels[idx]
            return kw
    else:
        plot_kwargs_cb = plot_kwargs_cb or (lambda idx: {})
    if interpolate is None:
        interpolate = y.ndim == 3 and y.shape[1] > 1
    if interpolate and y.ndim == 3:
        _y = y[:, 0, :]
    else:
        _y = y
    if indices is None:
        indices = range(_y.shape[-1])
    if lines is None:
        lines = interpolate in (None, False)
    markers = len(x) < m_lim
    if yerr is not None:
        for idx in indices:
            clr = plot_kwargs_cb(idx)['c']
            ax.fill_between(x, _y[:, idx] - yerr[:, idx], _y[:, idx] + yerr[:, idx], facecolor=clr, alpha=0.3)
    if isinstance(yscale, str) and 'linthreshy' in yscale:
        arg, kw = yscale.split(';')
        thresh = eval('dict(%s)' % kw)['linthreshy']
        ax.axhline(thresh, linewidth=0.5, linestyle='--', color='k', alpha=0.5)
        ax.axhline(-thresh, linewidth=0.5, linestyle='--', color='k', alpha=0.5)
    if labels is None:
        labels = names if latex_names is None else [tex_lbl_fmt % ln.strip('$') for ln in latex_names]
    for idx in indices:
        ax.plot(x, _y[:, idx], **plot_kwargs_cb(idx, lines=lines, labels=labels))
        if markers:
            ax.plot(x, _y[:, idx], **plot_kwargs_cb(idx, lines=False, markers=markers, labels=labels))
    if xlabel is None:
        try:
            ax.set_xlabel(_latex_from_dimensionality(x.dimensionality))
        except AttributeError:
            pass
    else:
        ax.set_xlabel(xlabel)
    if ylabel is None:
        try:
            ax.set_ylabel(_latex_from_dimensionality(_y.dimensionality))
        except AttributeError:
            pass
    else:
        ax.set_ylabel(ylabel)
    if interpolate:
        if interpolate is True:
            interpolate = 20
        if isinstance(interpolate, int):
            if interpolate > 0:
                x_plot = np.concatenate([np.linspace(a, b, interpolate) for a, b in zip(x[:-1], x[1:])])
            elif interpolate < 0:
                x_plot = np.concatenate([np.logspace(np.log10(a), np.log10(b), -interpolate) for a, b in zip(x[:-1], x[1:])])
        else:
            x_plot = interpolate
        if interp_from_deriv is None:
            import scipy.interpolate
            interp_from_deriv = scipy.interpolate.BPoly.from_derivatives
        y2 = np.empty((x_plot.size, _y.shape[-1]))
        for idx in range(_y.shape[-1]):
            interp_cb = interp_from_deriv(x, y[..., idx])
            y2[:, idx] = interp_cb(x_plot)
        for idx in indices:
            ax.plot(x_plot, y2[:, idx], **plot_kwargs_cb(idx, lines=True, markers=False))
        return (x_plot, y2)
    if xscale is not None:
        _set_scale(ax.set_xscale, xscale)
    if yscale is not None:
        _set_scale(ax.set_yscale, yscale)
    if legend is True:
        ax.legend()
    elif legend in (None, False):
        pass
    else:
        ax.legend(**legend)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    return ax