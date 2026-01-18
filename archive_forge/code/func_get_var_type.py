from __future__ import annotations
import typing
import numpy as np
from .._utils import array_kind
def get_var_type(col: pd.Series) -> Literal['c', 'o', 'u']:
    """
    Return var_type (for KDEMultivariate) of the column

    Parameters
    ----------
    col :
        A dataframe column.

    Returns
    -------
    out :
        Character that denotes the type of column.
        `c` for continuous, `o` for ordered categorical and
        `u` for unordered categorical or if not sure.

    See Also
    --------
    statsmodels.nonparametric.kernel_density.KDEMultivariate : For the origin
        of the character codes.
    """
    if array_kind.continuous(col):
        return 'c'
    elif array_kind.discrete(col):
        return 'o' if array_kind.ordinal else 'u'
    else:
        return 'u'