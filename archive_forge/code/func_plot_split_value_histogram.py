import math
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .basic import Booster, _data_from_pandas, _is_zero, _log_warning, _MissingType
from .compat import GRAPHVIZ_INSTALLED, MATPLOTLIB_INSTALLED, pd_DataFrame
from .sklearn import LGBMModel
def plot_split_value_histogram(booster: Union[Booster, LGBMModel], feature: Union[int, str], bins: Union[int, str, None]=None, ax=None, width_coef: float=0.8, xlim: Optional[Tuple[float, float]]=None, ylim: Optional[Tuple[float, float]]=None, title: Optional[str]='Split value histogram for feature with @index/name@ @feature@', xlabel: Optional[str]='Feature split value', ylabel: Optional[str]='Count', figsize: Optional[Tuple[float, float]]=None, dpi: Optional[int]=None, grid: bool=True, **kwargs: Any) -> Any:
    """Plot split value histogram for the specified feature of the model.

    Parameters
    ----------
    booster : Booster or LGBMModel
        Booster or LGBMModel instance of which feature split value histogram should be plotted.
    feature : int or str
        The feature name or index the histogram is plotted for.
        If int, interpreted as index.
        If str, interpreted as name.
    bins : int, str or None, optional (default=None)
        The maximum number of bins.
        If None, the number of bins equals number of unique split values.
        If str, it should be one from the list of the supported values by ``numpy.histogram()`` function.
    ax : matplotlib.axes.Axes or None, optional (default=None)
        Target axes instance.
        If None, new figure and axes will be created.
    width_coef : float, optional (default=0.8)
        Coefficient for histogram bar width.
    xlim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.xlim()``.
    ylim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.ylim()``.
    title : str or None, optional (default="Split value histogram for feature with @index/name@ @feature@")
        Axes title.
        If None, title is disabled.
        @feature@ placeholder can be used, and it will be replaced with the value of ``feature`` parameter.
        @index/name@ placeholder can be used,
        and it will be replaced with ``index`` word in case of ``int`` type ``feature`` parameter
        or ``name`` word in case of ``str`` type ``feature`` parameter.
    xlabel : str or None, optional (default="Feature split value")
        X-axis title label.
        If None, title is disabled.
    ylabel : str or None, optional (default="Count")
        Y-axis title label.
        If None, title is disabled.
    figsize : tuple of 2 elements or None, optional (default=None)
        Figure size.
    dpi : int or None, optional (default=None)
        Resolution of the figure.
    grid : bool, optional (default=True)
        Whether to add a grid for axes.
    **kwargs
        Other parameters passed to ``ax.bar()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with specified model's feature split value histogram.
    """
    if MATPLOTLIB_INSTALLED:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    else:
        raise ImportError('You must install matplotlib and restart your session to plot split value histogram.')
    if isinstance(booster, LGBMModel):
        booster = booster.booster_
    elif not isinstance(booster, Booster):
        raise TypeError('booster must be Booster or LGBMModel.')
    hist, split_bins = booster.get_split_value_histogram(feature=feature, bins=bins, xgboost_style=False)
    if np.count_nonzero(hist) == 0:
        raise ValueError(f'Cannot plot split value histogram, because feature {feature} was not used in splitting')
    width = width_coef * (split_bins[1] - split_bins[0])
    centred = (split_bins[:-1] + split_bins[1:]) / 2
    if ax is None:
        if figsize is not None:
            _check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.bar(centred, hist, align='center', width=width, **kwargs)
    if xlim is not None:
        _check_not_tuple_of_2_elements(xlim, 'xlim')
    else:
        range_result = split_bins[-1] - split_bins[0]
        xlim = (split_bins[0] - range_result * 0.2, split_bins[-1] + range_result * 0.2)
    ax.set_xlim(xlim)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if ylim is not None:
        _check_not_tuple_of_2_elements(ylim, 'ylim')
    else:
        ylim = (0, max(hist) * 1.1)
    ax.set_ylim(ylim)
    if title is not None:
        title = title.replace('@feature@', str(feature))
        title = title.replace('@index/name@', 'name' if isinstance(feature, str) else 'index')
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax