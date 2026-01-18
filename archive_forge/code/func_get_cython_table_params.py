from __future__ import annotations
from decimal import Decimal
import operator
import os
from sys import byteorder
from typing import (
import warnings
import numpy as np
from pandas._config.localization import (
from pandas.compat import pa_version_under10p1
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
from pandas._testing._io import (
from pandas._testing._warnings import (
from pandas._testing.asserters import (
from pandas._testing.compat import (
from pandas._testing.contexts import (
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import extract_array
def get_cython_table_params(ndframe, func_names_and_expected):
    """
    Combine frame, functions from com._cython_table
    keys and expected result.

    Parameters
    ----------
    ndframe : DataFrame or Series
    func_names_and_expected : Sequence of two items
        The first item is a name of a NDFrame method ('sum', 'prod') etc.
        The second item is the expected return value.

    Returns
    -------
    list
        List of three items (DataFrame, function, expected result)
    """
    results = []
    for func_name, expected in func_names_and_expected:
        results.append((ndframe, func_name, expected))
        results += [(ndframe, func, expected) for func, name in cython_table if name == func_name]
    return results