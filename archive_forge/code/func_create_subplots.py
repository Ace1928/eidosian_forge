from __future__ import annotations
from math import ceil
from typing import TYPE_CHECKING
import warnings
from matplotlib import ticker
import matplotlib.table
import numpy as np
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
def create_subplots(naxes: int, sharex: bool=False, sharey: bool=False, squeeze: bool=True, subplot_kw=None, ax=None, layout=None, layout_type: str='box', **fig_kw):
    """
    Create a figure with a set of subplots already made.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Parameters
    ----------
    naxes : int
      Number of required axes. Exceeded axes are set invisible. Default is
      nrows * ncols.

    sharex : bool
      If True, the X axis will be shared amongst all subplots.

    sharey : bool
      If True, the Y axis will be shared amongst all subplots.

    squeeze : bool

      If True, extra dimensions are squeezed out from the returned axis object:
        - if only one subplot is constructed (nrows=ncols=1), the resulting
        single Axis object is returned as a scalar.
        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy object
        array of Axis objects are returned as numpy 1-d arrays.
        - for NxM subplots with N>1 and M>1 are returned as a 2d array.

      If False, no squeezing is done: the returned axis object is always
      a 2-d array containing Axis instances, even if it ends up being 1x1.

    subplot_kw : dict
      Dict with keywords passed to the add_subplot() call used to create each
      subplots.

    ax : Matplotlib axis object, optional

    layout : tuple
      Number of rows and columns of the subplot grid.
      If not specified, calculated from naxes and layout_type

    layout_type : {'box', 'horizontal', 'vertical'}, default 'box'
      Specify how to layout the subplot grid.

    fig_kw : Other keyword arguments to be passed to the figure() call.
        Note that all keywords not recognized above will be
        automatically included here.

    Returns
    -------
    fig, ax : tuple
      - fig is the Matplotlib Figure object
      - ax can be either a single axis object or an array of axis objects if
      more than one subplot was created.  The dimensions of the resulting array
      can be controlled with the squeeze keyword, see above.

    Examples
    --------
    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    # Just a figure and one subplot
    f, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # Two subplots, unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Four polar axes
    plt.subplots(2, 2, subplot_kw=dict(polar=True))
    """
    import matplotlib.pyplot as plt
    if subplot_kw is None:
        subplot_kw = {}
    if ax is None:
        fig = plt.figure(**fig_kw)
    else:
        if is_list_like(ax):
            if squeeze:
                ax = flatten_axes(ax)
            if layout is not None:
                warnings.warn('When passing multiple axes, layout keyword is ignored.', UserWarning, stacklevel=find_stack_level())
            if sharex or sharey:
                warnings.warn('When passing multiple axes, sharex and sharey are ignored. These settings must be specified when creating axes.', UserWarning, stacklevel=find_stack_level())
            if ax.size == naxes:
                fig = ax.flat[0].get_figure()
                return (fig, ax)
            else:
                raise ValueError(f'The number of passed axes must be {naxes}, the same as the output plot')
        fig = ax.get_figure()
        if naxes == 1:
            if squeeze:
                return (fig, ax)
            else:
                return (fig, flatten_axes(ax))
        else:
            warnings.warn('To output multiple subplots, the figure containing the passed axes is being cleared.', UserWarning, stacklevel=find_stack_level())
            fig.clear()
    nrows, ncols = _get_layout(naxes, layout=layout, layout_type=layout_type)
    nplots = nrows * ncols
    axarr = np.empty(nplots, dtype=object)
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)
    if sharex:
        subplot_kw['sharex'] = ax0
    if sharey:
        subplot_kw['sharey'] = ax0
    axarr[0] = ax0
    for i in range(1, nplots):
        kwds = subplot_kw.copy()
        if i >= naxes:
            kwds['sharex'] = None
            kwds['sharey'] = None
        ax = fig.add_subplot(nrows, ncols, i + 1, **kwds)
        axarr[i] = ax
    if naxes != nplots:
        for ax in axarr[naxes:]:
            ax.set_visible(False)
    handle_shared_axes(axarr, nplots, naxes, nrows, ncols, sharex, sharey)
    if squeeze:
        if nplots == 1:
            axes = axarr[0]
        else:
            axes = axarr.reshape(nrows, ncols).squeeze()
    else:
        axes = axarr.reshape(nrows, ncols)
    return (fig, axes)