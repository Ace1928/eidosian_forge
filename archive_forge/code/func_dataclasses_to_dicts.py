from __future__ import annotations
from collections import abc
from typing import (
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core import (
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.indexes.api import (
from pandas.core.internals.array_manager import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def dataclasses_to_dicts(data):
    """
    Converts a list of dataclass instances to a list of dictionaries.

    Parameters
    ----------
    data : List[Type[dataclass]]

    Returns
    --------
    list_dict : List[dict]

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: int
    ...     y: int

    >>> dataclasses_to_dicts([Point(1, 2), Point(2, 3)])
    [{'x': 1, 'y': 2}, {'x': 2, 'y': 3}]

    """
    from dataclasses import asdict
    return list(map(asdict, data))