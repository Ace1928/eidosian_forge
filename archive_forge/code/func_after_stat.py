from __future__ import annotations
import numbers
import typing
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from ..exceptions import PlotnineError
def after_stat(x):
    """
    Evaluate mapping after statistic has been calculated

    Parameters
    ----------
    x : str
        An expression

    See Also
    --------
    plotnine.after_scale
    plotnine.stage
    """
    return stage(after_stat=x)