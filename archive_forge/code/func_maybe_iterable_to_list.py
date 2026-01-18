from __future__ import annotations
import builtins
from collections import (
from collections.abc import (
import contextlib
from functools import partial
import inspect
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.compat.numpy import np_version_gte1p24
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import iterable_not_string
def maybe_iterable_to_list(obj: Iterable[T] | T) -> Collection[T] | T:
    """
    If obj is Iterable but not list-like, consume into list.
    """
    if isinstance(obj, abc.Iterable) and (not isinstance(obj, abc.Sized)):
        return list(obj)
    obj = cast(Collection, obj)
    return obj