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
def is_data_like(obj: Any) -> TypeGuard[DataLike]:
    """
    Return True if obj could be data

    Parameters
    ----------
    obj : object
        Object that could be data

    Returns
    -------
    out : bool
        Whether obj could represent data as expected by
        ggplot(), geom() or stat().
    """
    return isinstance(obj, pd.DataFrame) or callable(obj) or hasattr(obj, 'to_pandas')