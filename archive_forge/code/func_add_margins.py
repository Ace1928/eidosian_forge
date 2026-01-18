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
def add_margins(df: pd.DataFrame, vars: tuple[Sequence[str], Sequence[str]], margins: bool | Sequence[str]=True) -> pd.DataFrame:
    """
    Add margins to a data frame.

    All margining variables will be converted to factors.

    Parameters
    ----------
    df : dataframe
        input data frame

    vars : list
        a list of 2 lists | tuples vectors giving the
        variables in each dimension

    margins : bool | list
        variable names to compute margins for.
        True will compute all possible margins.
    """
    margin_vars = _margins(vars, margins)
    if not margin_vars:
        return df
    margin_dfs = [df]
    for vlst in margin_vars[1:]:
        dfx = df.copy()
        for v in vlst:
            dfx[v] = '(all)'
        margin_dfs.append(dfx)
    merged = pd.concat(margin_dfs, axis=0)
    merged.reset_index(drop=True, inplace=True)
    categories = {}
    for v in itertools.chain(*vars):
        col = df[v]
        if not isinstance(df[v], pd.Categorical):
            col = pd.Categorical(df[v])
        categories[v] = col.categories
        if '(all)' not in categories[v]:
            categories[v] = categories[v].insert(len(categories[v]), '(all)')
    for v in merged.columns.intersection(list(categories.keys())):
        merged[v] = merged[v].astype(pd.CategoricalDtype(categories[v]))
    return merged