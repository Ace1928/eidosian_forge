import math
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .basic import Booster, _data_from_pandas, _is_zero, _log_warning, _MissingType
from .compat import GRAPHVIZ_INSTALLED, MATPLOTLIB_INSTALLED, pd_DataFrame
from .sklearn import LGBMModel
def plot_importance(booster: Union[Booster, LGBMModel], ax=None, height: float=0.2, xlim: Optional[Tuple[float, float]]=None, ylim: Optional[Tuple[float, float]]=None, title: Optional[str]='Feature importance', xlabel: Optional[str]='Feature importance', ylabel: Optional[str]='Features', importance_type: str='auto', max_num_features: Optional[int]=None, ignore_zero: bool=True, figsize: Optional[Tuple[float, float]]=None, dpi: Optional[int]=None, grid: bool=True, precision: Optional[int]=3, **kwargs: Any) -> Any:
    """Plot model's feature importances.

    Parameters
    ----------
    booster : Booster or LGBMModel
        Booster or LGBMModel instance which feature importance should be plotted.
    ax : matplotlib.axes.Axes or None, optional (default=None)
        Target axes instance.
        If None, new figure and axes will be created.
    height : float, optional (default=0.2)
        Bar height, passed to ``ax.barh()``.
    xlim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.xlim()``.
    ylim : tuple of 2 elements or None, optional (default=None)
        Tuple passed to ``ax.ylim()``.
    title : str or None, optional (default="Feature importance")
        Axes title.
        If None, title is disabled.
    xlabel : str or None, optional (default="Feature importance")
        X-axis title label.
        If None, title is disabled.
        @importance_type@ placeholder can be used, and it will be replaced with the value of ``importance_type`` parameter.
    ylabel : str or None, optional (default="Features")
        Y-axis title label.
        If None, title is disabled.
    importance_type : str, optional (default="auto")
        How the importance is calculated.
        If "auto", if ``booster`` parameter is LGBMModel, ``booster.importance_type`` attribute is used; "split" otherwise.
        If "split", result contains numbers of times the feature is used in a model.
        If "gain", result contains total gains of splits which use the feature.
    max_num_features : int or None, optional (default=None)
        Max number of top features displayed on plot.
        If None or <1, all features will be displayed.
    ignore_zero : bool, optional (default=True)
        Whether to ignore features with zero importance.
    figsize : tuple of 2 elements or None, optional (default=None)
        Figure size.
    dpi : int or None, optional (default=None)
        Resolution of the figure.
    grid : bool, optional (default=True)
        Whether to add a grid for axes.
    precision : int or None, optional (default=3)
        Used to restrict the display of floating point values to a certain precision.
    **kwargs
        Other parameters passed to ``ax.barh()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with model's feature importances.
    """
    if MATPLOTLIB_INSTALLED:
        import matplotlib.pyplot as plt
    else:
        raise ImportError('You must install matplotlib and restart your session to plot importance.')
    if isinstance(booster, LGBMModel):
        if importance_type == 'auto':
            importance_type = booster.importance_type
        booster = booster.booster_
    elif isinstance(booster, Booster):
        if importance_type == 'auto':
            importance_type = 'split'
    else:
        raise TypeError('booster must be Booster or LGBMModel.')
    importance = booster.feature_importance(importance_type=importance_type)
    feature_name = booster.feature_name()
    if not len(importance):
        raise ValueError("Booster's feature_importance is empty.")
    tuples = sorted(zip(feature_name, importance), key=lambda x: x[1])
    if ignore_zero:
        tuples = [x for x in tuples if x[1] > 0]
    if max_num_features is not None and max_num_features > 0:
        tuples = tuples[-max_num_features:]
    labels, values = zip(*tuples)
    if ax is None:
        if figsize is not None:
            _check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)
    for x, y in zip(values, ylocs):
        ax.text(x + 1, y, _float2str(x, precision) if importance_type == 'gain' else x, va='center')
    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)
    if xlim is not None:
        _check_not_tuple_of_2_elements(xlim, 'xlim')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)
    if ylim is not None:
        _check_not_tuple_of_2_elements(ylim, 'ylim')
    else:
        ylim = (-1, len(values))
    ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        xlabel = xlabel.replace('@importance_type@', importance_type)
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax