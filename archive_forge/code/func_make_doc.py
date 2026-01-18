from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
def make_doc(name: str, ndim: int) -> str:
    """
    Generate the docstring for a Series/DataFrame reduction.
    """
    if ndim == 1:
        name1 = 'scalar'
        name2 = 'Series'
        axis_descr = '{index (0)}'
    else:
        name1 = 'Series'
        name2 = 'DataFrame'
        axis_descr = '{index (0), columns (1)}'
    if name == 'any':
        base_doc = _bool_doc
        desc = _any_desc
        see_also = _any_see_also
        examples = _any_examples
        kwargs = {'empty_value': 'False'}
    elif name == 'all':
        base_doc = _bool_doc
        desc = _all_desc
        see_also = _all_see_also
        examples = _all_examples
        kwargs = {'empty_value': 'True'}
    elif name == 'min':
        base_doc = _num_doc
        desc = 'Return the minimum of the values over the requested axis.\n\nIf you want the *index* of the minimum, use ``idxmin``. This is the equivalent of the ``numpy.ndarray`` method ``argmin``.'
        see_also = _stat_func_see_also
        examples = _min_examples
        kwargs = {'min_count': ''}
    elif name == 'max':
        base_doc = _num_doc
        desc = 'Return the maximum of the values over the requested axis.\n\nIf you want the *index* of the maximum, use ``idxmax``. This is the equivalent of the ``numpy.ndarray`` method ``argmax``.'
        see_also = _stat_func_see_also
        examples = _max_examples
        kwargs = {'min_count': ''}
    elif name == 'sum':
        base_doc = _sum_prod_doc
        desc = 'Return the sum of the values over the requested axis.\n\nThis is equivalent to the method ``numpy.sum``.'
        see_also = _stat_func_see_also
        examples = _sum_examples
        kwargs = {'min_count': _min_count_stub}
    elif name == 'prod':
        base_doc = _sum_prod_doc
        desc = 'Return the product of the values over the requested axis.'
        see_also = _stat_func_see_also
        examples = _prod_examples
        kwargs = {'min_count': _min_count_stub}
    elif name == 'median':
        base_doc = _num_doc
        desc = 'Return the median of the values over the requested axis.'
        see_also = ''
        examples = "\n\n            Examples\n            --------\n            >>> s = pd.Series([1, 2, 3])\n            >>> s.median()\n            2.0\n\n            With a DataFrame\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])\n            >>> df\n                   a   b\n            tiger  1   2\n            zebra  2   3\n            >>> df.median()\n            a   1.5\n            b   2.5\n            dtype: float64\n\n            Using axis=1\n\n            >>> df.median(axis=1)\n            tiger   1.5\n            zebra   2.5\n            dtype: float64\n\n            In this case, `numeric_only` should be set to `True`\n            to avoid getting an error.\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},\n            ...                   index=['tiger', 'zebra'])\n            >>> df.median(numeric_only=True)\n            a   1.5\n            dtype: float64"
        kwargs = {'min_count': ''}
    elif name == 'mean':
        base_doc = _num_doc
        desc = 'Return the mean of the values over the requested axis.'
        see_also = ''
        examples = "\n\n            Examples\n            --------\n            >>> s = pd.Series([1, 2, 3])\n            >>> s.mean()\n            2.0\n\n            With a DataFrame\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])\n            >>> df\n                   a   b\n            tiger  1   2\n            zebra  2   3\n            >>> df.mean()\n            a   1.5\n            b   2.5\n            dtype: float64\n\n            Using axis=1\n\n            >>> df.mean(axis=1)\n            tiger   1.5\n            zebra   2.5\n            dtype: float64\n\n            In this case, `numeric_only` should be set to `True` to avoid\n            getting an error.\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},\n            ...                   index=['tiger', 'zebra'])\n            >>> df.mean(numeric_only=True)\n            a   1.5\n            dtype: float64"
        kwargs = {'min_count': ''}
    elif name == 'var':
        base_doc = _num_ddof_doc
        desc = 'Return unbiased variance over requested axis.\n\nNormalized by N-1 by default. This can be changed using the ddof argument.'
        examples = _var_examples
        see_also = ''
        kwargs = {'notes': ''}
    elif name == 'std':
        base_doc = _num_ddof_doc
        desc = 'Return sample standard deviation over requested axis.\n\nNormalized by N-1 by default. This can be changed using the ddof argument.'
        examples = _std_examples
        see_also = ''
        kwargs = {'notes': _std_notes}
    elif name == 'sem':
        base_doc = _num_ddof_doc
        desc = 'Return unbiased standard error of the mean over requested axis.\n\nNormalized by N-1 by default. This can be changed using the ddof argument'
        examples = "\n\n            Examples\n            --------\n            >>> s = pd.Series([1, 2, 3])\n            >>> s.sem().round(6)\n            0.57735\n\n            With a DataFrame\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])\n            >>> df\n                   a   b\n            tiger  1   2\n            zebra  2   3\n            >>> df.sem()\n            a   0.5\n            b   0.5\n            dtype: float64\n\n            Using axis=1\n\n            >>> df.sem(axis=1)\n            tiger   0.5\n            zebra   0.5\n            dtype: float64\n\n            In this case, `numeric_only` should be set to `True`\n            to avoid getting an error.\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']},\n            ...                   index=['tiger', 'zebra'])\n            >>> df.sem(numeric_only=True)\n            a   0.5\n            dtype: float64"
        see_also = ''
        kwargs = {'notes': ''}
    elif name == 'skew':
        base_doc = _num_doc
        desc = 'Return unbiased skew over requested axis.\n\nNormalized by N-1.'
        see_also = ''
        examples = "\n\n            Examples\n            --------\n            >>> s = pd.Series([1, 2, 3])\n            >>> s.skew()\n            0.0\n\n            With a DataFrame\n\n            >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [1, 3, 5]},\n            ...                   index=['tiger', 'zebra', 'cow'])\n            >>> df\n                    a   b   c\n            tiger   1   2   1\n            zebra   2   3   3\n            cow     3   4   5\n            >>> df.skew()\n            a   0.0\n            b   0.0\n            c   0.0\n            dtype: float64\n\n            Using axis=1\n\n            >>> df.skew(axis=1)\n            tiger   1.732051\n            zebra  -1.732051\n            cow     0.000000\n            dtype: float64\n\n            In this case, `numeric_only` should be set to `True` to avoid\n            getting an error.\n\n            >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['T', 'Z', 'X']},\n            ...                   index=['tiger', 'zebra', 'cow'])\n            >>> df.skew(numeric_only=True)\n            a   0.0\n            dtype: float64"
        kwargs = {'min_count': ''}
    elif name == 'kurt':
        base_doc = _num_doc
        desc = "Return unbiased kurtosis over requested axis.\n\nKurtosis obtained using Fisher's definition of\nkurtosis (kurtosis of normal == 0.0). Normalized by N-1."
        see_also = ''
        examples = "\n\n            Examples\n            --------\n            >>> s = pd.Series([1, 2, 2, 3], index=['cat', 'dog', 'dog', 'mouse'])\n            >>> s\n            cat    1\n            dog    2\n            dog    2\n            mouse  3\n            dtype: int64\n            >>> s.kurt()\n            1.5\n\n            With a DataFrame\n\n            >>> df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [3, 4, 4, 4]},\n            ...                   index=['cat', 'dog', 'dog', 'mouse'])\n            >>> df\n                   a   b\n              cat  1   3\n              dog  2   4\n              dog  2   4\n            mouse  3   4\n            >>> df.kurt()\n            a   1.5\n            b   4.0\n            dtype: float64\n\n            With axis=None\n\n            >>> df.kurt(axis=None).round(6)\n            -0.988693\n\n            Using axis=1\n\n            >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [3, 4], 'd': [1, 2]},\n            ...                   index=['cat', 'dog'])\n            >>> df.kurt(axis=1)\n            cat   -6.0\n            dog   -6.0\n            dtype: float64"
        kwargs = {'min_count': ''}
    elif name == 'cumsum':
        base_doc = _cnum_doc
        desc = 'sum'
        see_also = ''
        examples = _cumsum_examples
        kwargs = {'accum_func_name': 'sum'}
    elif name == 'cumprod':
        base_doc = _cnum_doc
        desc = 'product'
        see_also = ''
        examples = _cumprod_examples
        kwargs = {'accum_func_name': 'prod'}
    elif name == 'cummin':
        base_doc = _cnum_doc
        desc = 'minimum'
        see_also = ''
        examples = _cummin_examples
        kwargs = {'accum_func_name': 'min'}
    elif name == 'cummax':
        base_doc = _cnum_doc
        desc = 'maximum'
        see_also = ''
        examples = _cummax_examples
        kwargs = {'accum_func_name': 'max'}
    else:
        raise NotImplementedError
    docstr = base_doc.format(desc=desc, name=name, name1=name1, name2=name2, axis_descr=axis_descr, see_also=see_also, examples=examples, **kwargs)
    return docstr