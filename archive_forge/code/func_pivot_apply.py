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
def pivot_apply(df, column, index, func, *args, **kwargs):
    """
    Apply a function to each group of a column

    The function is kind of equivalent to R's *tapply*.

    Parameters
    ----------
    df : dataframe
        Dataframe to be pivoted
    column : str
        Column to apply function to.
    index : str
        Column that will be grouped on (and whose unique values
        will make up the index of the returned dataframe)
    func : callable
        Function to apply to each column group. It *should* return
        a single value.
    *args : tuple
        Arguments to `func`
    **kwargs : dict
        Keyword arguments to `func`

    Returns
    -------
    out : dataframe
        Dataframe with index `index` and column `column` of
        computed/aggregate values .
    """

    def _func(x):
        return func(x, *args, **kwargs)
    return df.pivot_table(column, index, aggfunc=_func)[column]