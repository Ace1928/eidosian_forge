from __future__ import annotations
import itertools
import types
import typing
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from .._utils import cross_join, match
from ..exceptions import PlotnineError
from ..scales.scales import Scales
from .strips import Strips
def set_limits_breaks_and_labels(self, panel_params: panel_view, ax: Axes):
    """
        Add limits, breaks and labels to the axes

        Parameters
        ----------
        panel_params :
            range information for the axes
        ax :
            Axes
        """
    from .._mpl.ticker import MyFixedFormatter

    def _inf_to_none(t: tuple[float, float]) -> tuple[float | None, float | None]:
        """
            Replace infinities with None
            """
        a = t[0] if np.isfinite(t[0]) else None
        b = t[1] if np.isfinite(t[1]) else None
        return (a, b)
    theme = self.theme
    ax.set_xlim(*_inf_to_none(panel_params.x.range))
    ax.set_ylim(*_inf_to_none(panel_params.y.range))
    if typing.TYPE_CHECKING:
        assert callable(ax.set_xticks)
        assert callable(ax.set_yticks)
    ax.set_xticks(panel_params.x.breaks, panel_params.x.labels)
    ax.set_yticks(panel_params.y.breaks, panel_params.y.labels)
    ax.set_xticks(panel_params.x.minor_breaks, minor=True)
    ax.set_yticks(panel_params.y.minor_breaks, minor=True)
    ax.xaxis.set_major_formatter(MyFixedFormatter(panel_params.x.labels))
    ax.yaxis.set_major_formatter(MyFixedFormatter(panel_params.y.labels))
    margin = theme.getp(('axis_text_x', 'margin'))
    pad_x = margin.get_as('t', 'pt')
    margin = theme.getp(('axis_text_y', 'margin'))
    pad_y = margin.get_as('r', 'pt')
    ax.tick_params(axis='x', which='major', pad=pad_x)
    ax.tick_params(axis='y', which='major', pad=pad_y)