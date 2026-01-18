from itertools import product
from math import ceil, floor, sqrt
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union, no_type_check
import numpy as np
import torch
from torch import Tensor
from torchmetrics.utilities.imports import _LATEX_AVAILABLE, _MATPLOTLIB_AVAILABLE, _SCIENCEPLOT_AVAILABLE
@style_change(_style)
def plot_single_or_multi_val(val: Union[Tensor, Sequence[Tensor], Dict[str, Tensor], Sequence[Dict[str, Tensor]]], ax: Optional[_AX_TYPE]=None, higher_is_better: Optional[bool]=None, lower_bound: Optional[float]=None, upper_bound: Optional[float]=None, legend_name: Optional[str]=None, name: Optional[str]=None) -> _PLOT_OUT_TYPE:
    """Plot a single metric value or multiple, including bounds of value if existing.

    Args:
        val: A single tensor with one or multiple values (multiclass/label/output format) or a list of such tensors.
            If a list is provided the values are interpreted as a time series of evolving values.
        ax: Axis from a figure.
        higher_is_better: Indicates if a label indicating where the optimal value it should be added to the figure
        lower_bound: lower value that the metric can take
        upper_bound: upper value that the metric can take
        legend_name: for class based metrics specify the legend prefix e.g. Class or Label to use when multiple values
            are provided
        name: Name of the metric to use for the y-axis label

    Returns:
        A tuple consisting of the figure and respective ax objects of the generated figure

    Raises:
        ModuleNotFoundError:
            If `matplotlib` is not installed

    """
    _error_on_missing_matplotlib()
    fig, ax = plt.subplots() if ax is None else (None, ax)
    ax.get_xaxis().set_visible(False)
    if isinstance(val, Tensor):
        if val.numel() == 1:
            ax.plot([val.detach().cpu()], marker='o', markersize=10)
        else:
            for i, v in enumerate(val):
                label = f'{legend_name} {i}' if legend_name else f'{i}'
                ax.plot(i, v.detach().cpu(), marker='o', markersize=10, linestyle='None', label=label)
    elif isinstance(val, dict):
        for i, (k, v) in enumerate(val.items()):
            if v.numel() != 1:
                ax.plot(v.detach().cpu(), marker='o', markersize=10, linestyle='-', label=k)
                ax.get_xaxis().set_visible(True)
                ax.set_xlabel('Step')
                ax.set_xticks(torch.arange(len(v)))
            else:
                ax.plot(i, v.detach().cpu(), marker='o', markersize=10, label=k)
    elif isinstance(val, Sequence):
        n_steps = len(val)
        if isinstance(val[0], dict):
            val = {k: torch.stack([val[i][k] for i in range(n_steps)]) for k in val[0]}
            for k, v in val.items():
                ax.plot(v.detach().cpu(), marker='o', markersize=10, linestyle='-', label=k)
        else:
            val = torch.stack(val, 0)
            multi_series = val.ndim != 1
            val = val.T if multi_series else val.unsqueeze(0)
            for i, v in enumerate(val):
                label = (f'{legend_name} {i}' if legend_name else f'{i}') if multi_series else ''
                ax.plot(v.detach().cpu(), marker='o', markersize=10, linestyle='-', label=label)
        ax.get_xaxis().set_visible(True)
        ax.set_xlabel('Step')
        ax.set_xticks(torch.arange(n_steps))
    else:
        raise ValueError('Got unknown format for argument `val`.')
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
    ylim = ax.get_ylim()
    if lower_bound is not None and upper_bound is not None:
        factor = 0.1 * (upper_bound - lower_bound)
    else:
        factor = 0.1 * (ylim[1] - ylim[0])
    ax.set_ylim(bottom=lower_bound - factor if lower_bound is not None else ylim[0] - factor, top=upper_bound + factor if upper_bound is not None else ylim[1] + factor)
    ax.grid(True)
    ax.set_ylabel(name if name is not None else None)
    xlim = ax.get_xlim()
    factor = 0.1 * (xlim[1] - xlim[0])
    y_lines = []
    if lower_bound is not None:
        y_lines.append(lower_bound)
    if upper_bound is not None:
        y_lines.append(upper_bound)
    ax.hlines(y_lines, xlim[0], xlim[1], linestyles='dashed', colors='k')
    if higher_is_better is not None:
        if lower_bound is not None and (not higher_is_better):
            ax.set_xlim(xlim[0] - factor, xlim[1])
            ax.text(xlim[0], lower_bound, s='Optimal \n value', horizontalalignment='center', verticalalignment='center')
        if upper_bound is not None and higher_is_better:
            ax.set_xlim(xlim[0] - factor, xlim[1])
            ax.text(xlim[0], upper_bound, s='Optimal \n value', horizontalalignment='center', verticalalignment='center')
    return (fig, ax)