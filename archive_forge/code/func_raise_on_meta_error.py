from __future__ import annotations
import math
import re
import sys
import textwrap
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from numbers import Number
from typing import TypeVar, overload
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
import dask
from dask.base import get_scheduler, is_dask_collection
from dask.core import get_deps
from dask.dataframe import (  # noqa: F401 register pandas extension types
from dask.dataframe._compat import PANDAS_GE_150, tm  # noqa: F401
from dask.dataframe.dispatch import (  # noqa : F401
from dask.dataframe.extensions import make_scalar
from dask.typing import NoDefault, no_default
from dask.utils import (
@contextmanager
def raise_on_meta_error(funcname=None, udf=False):
    """Reraise errors in this block to show metadata inference failure.

    Parameters
    ----------
    funcname : str, optional
        If provided, will be added to the error message to indicate the
        name of the method that failed.
    """
    try:
        yield
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = ''.join(traceback.format_tb(exc_traceback))
        msg = 'Metadata inference failed{0}.\n\n'
        if udf:
            msg += 'You have supplied a custom function and Dask is unable to \ndetermine the type of output that that function returns. \n\nTo resolve this please provide a meta= keyword.\nThe docstring of the Dask function you ran should have more information.\n\n'
        msg += 'Original error is below:\n------------------------\n{1}\n\nTraceback:\n---------\n{2}'
        msg = msg.format(f' in `{funcname}`' if funcname else '', repr(e), tb)
        raise ValueError(msg) from e