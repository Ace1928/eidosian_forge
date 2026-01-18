from __future__ import annotations
import numbers
import typing
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from ..exceptions import PlotnineError
def after_scale(x):
    """
    Evaluate mapping after variable has been mapped to the scale

    This gives the user a chance to alter the value of a variable
    in the final units of the scale e.g. the rgb hex color.

    Parameters
    ----------
    x : str
        An expression

    See Also
    --------
    plotnine.after_stat
    plotnine.stage
    """
    return stage(after_scale=x)