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
def multitype_sort(arr: AnyArrayLike) -> list[Any]:
    """
    Sort elements of multiple types

    x is assumed to contain elements of different types, such that
    plain sort would raise a `TypeError`.

    Parameters
    ----------
    a : array_like
        Array of items to be sorted

    Returns
    -------
    out : list
        Items sorted within their type groups.
    """
    types = defaultdict(list)
    for x in arr:
        if isinstance(x, (int, float, complex)):
            types['number'].append(x)
        else:
            types[type(x)].append(x)
    for t, values in types.items():
        types[t] = sorted(values)
    return list(itertools.chain.from_iterable((types[t] for t in types)))