import abc
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._libs.tslibs import to_offset
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype
from pandas.core.resample import _get_timestamp_range_edges
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
@wraps(f)
def run_f_on_minimally_updated_metadata(self, *args, **kwargs):
    from .dataframe import PandasDataframe
    for obj in [self] + [o for o in args if isinstance(o, PandasDataframe)] + [v for v in kwargs.values() if isinstance(v, PandasDataframe)] + [d for o in args if isinstance(o, list) for d in o if isinstance(d, PandasDataframe)] + [d for _, o in kwargs.items() if isinstance(o, list) for d in o if isinstance(d, PandasDataframe)]:
        if apply_axis == 'both':
            if obj._deferred_index and obj._deferred_column:
                obj._propagate_index_objs(axis=None)
            elif obj._deferred_index:
                obj._propagate_index_objs(axis=0)
            elif obj._deferred_column:
                obj._propagate_index_objs(axis=1)
        elif apply_axis == 'opposite':
            if 'axis' not in kwargs:
                axis = args[axis_arg]
            else:
                axis = kwargs['axis']
            if axis == 0 and obj._deferred_column:
                obj._propagate_index_objs(axis=1)
            elif axis == 1 and obj._deferred_index:
                obj._propagate_index_objs(axis=0)
        elif apply_axis == 'rows':
            obj._propagate_index_objs(axis=0)
    result = f(self, *args, **kwargs)
    if apply_axis is None and (not transpose):
        result._deferred_index = self._deferred_index
        result._deferred_column = self._deferred_column
    elif apply_axis is None and transpose:
        result._deferred_index = self._deferred_column
        result._deferred_column = self._deferred_index
    elif apply_axis == 'opposite':
        if axis == 0:
            result._deferred_index = self._deferred_index
        else:
            result._deferred_column = self._deferred_column
    elif apply_axis == 'rows':
        result._deferred_column = self._deferred_column
    return result