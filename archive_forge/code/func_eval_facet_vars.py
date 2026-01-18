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
def eval_facet_vars(data: pd.DataFrame, vars: Sequence[str], env: Environment) -> pd.DataFrame:
    """
    Evaluate facet variables

    Parameters
    ----------
    data :
        Factet dataframe
    vars :
        Facet variables
    env :
        Plot environment

    Returns
    -------
    :
        Facet values that correspond to the specified
        variables.
    """

    def I(value: Any) -> Any:
        return value
    env = env.with_outer_namespace({'I': I})
    facet_vals = pd.DataFrame(index=data.index)
    for name in vars:
        if name in data:
            res = data[name]
        elif str.isidentifier(name):
            continue
        else:
            try:
                res = env.eval(name, inner_namespace=data)
            except NameError:
                continue
        facet_vals[name] = res
    return facet_vals