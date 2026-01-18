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
def combine_vars(data: list[pd.DataFrame], environment: Environment, vars: Sequence[str], drop: bool=True) -> pd.DataFrame:
    """
    Generate all combinations of data needed for facetting

    The first data frame in the list should be the default data
    for the plot. Other data frames in the list are ones that are
    added to the layers.
    """
    if len(vars) == 0:
        return pd.DataFrame()
    values = [eval_facet_vars(df, vars, environment) for df in data if df is not None]
    has_all = [x.shape[1] == len(vars) for x in values]
    if not any(has_all):
        raise PlotnineError('At least one layer must contain all variables used for facetting')
    base = pd.concat([x for i, x in enumerate(values) if has_all[i]], axis=0)
    base = base.drop_duplicates()
    if not drop:
        base = unique_combs(base)
    base = base.sort_values(base.columns.tolist())
    for i, value in enumerate(values):
        if has_all[i] or len(value.columns) == 0:
            continue
        old = base.loc[:, list(base.columns.difference(value.columns))]
        new = value.loc[:, list(base.columns.intersection(value.columns))].drop_duplicates()
        if not drop:
            new = unique_combs(new)
        base = pd.concat([base, cross_join(old, new)], ignore_index=True)
    if len(base) == 0:
        raise PlotnineError('Faceting variables must have at least one value')
    base = base.reset_index(drop=True)
    return base