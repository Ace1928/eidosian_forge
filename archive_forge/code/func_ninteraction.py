from __future__ import annotations
import inspect
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping import aes
def ninteraction(df: pd.DataFrame, drop: bool=False) -> list[int]:
    """
    Compute a unique numeric id for each unique row in
    a data frame. The ids start at 1 -- in the spirit
    of `plyr::id`

    Parameters
    ----------
    df : dataframe
        Rows
    drop : bool
        If true, drop unused categorical levels leaving no
        gaps in the assignments.

    Returns
    -------
    out : list
        Row asssignments.

    Notes
    -----
    So far there has been no need not to drop unused levels
    of categorical variables.
    """
    if len(df) == 0:
        return []
    if len(df.columns) == 1:
        return _id_var(df[df.columns[0]], drop)
    ids = df.apply(_id_var, axis=0)
    ids = ids.reindex(columns=list(reversed(ids.columns)))

    def len_unique(x):
        return len(np.unique(x))
    ndistinct: IntArray = ids.apply(len_unique, axis=0).to_numpy()
    combs = np.array(np.hstack([1, np.cumprod(ndistinct[:-1])]))
    mat = np.array(ids)
    res = (mat - 1) @ combs.T + 1
    res = np.array(res).flatten().tolist()
    if drop:
        return _id_var(res, drop)
    else:
        return list(res)