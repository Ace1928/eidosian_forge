from __future__ import annotations
import copy
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import NDFrameT
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
import pandas.core.algorithms as algos
from pandas.core.apply import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.generic import (
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import (
from pandas.tseries.frequencies import (
from pandas.tseries.offsets import (
def maybe_warn_args_and_kwargs(cls, kernel: str, args, kwargs) -> None:
    """
    Warn for deprecation of args and kwargs in resample functions.

    Parameters
    ----------
    cls : type
        Class to warn about.
    kernel : str
        Operation name.
    args : tuple or None
        args passed by user. Will be None if and only if kernel does not have args.
    kwargs : dict or None
        kwargs passed by user. Will be None if and only if kernel does not have kwargs.
    """
    warn_args = args is not None and len(args) > 0
    warn_kwargs = kwargs is not None and len(kwargs) > 0
    if warn_args and warn_kwargs:
        msg = 'args and kwargs'
    elif warn_args:
        msg = 'args'
    elif warn_kwargs:
        msg = 'kwargs'
    else:
        return
    warnings.warn(f'Passing additional {msg} to {cls.__name__}.{kernel} has no impact on the result and is deprecated. This will raise a TypeError in a future version of pandas.', category=FutureWarning, stacklevel=find_stack_level())