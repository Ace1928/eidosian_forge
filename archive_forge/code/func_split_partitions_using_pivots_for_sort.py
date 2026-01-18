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
@staticmethod
def split_partitions_using_pivots_for_sort(df: pandas.DataFrame, columns_info: 'list[ColumnInfo]', ascending: bool, closed_on_right: bool=True, **kwargs: dict) -> 'tuple[pandas.DataFrame, ...]':

    def add_attr(df, timestamp):
        if 'bin_bounds' in df.attrs:
            df.attrs['bin_bounds'] = (*df.attrs['bin_bounds'], timestamp)
        else:
            df.attrs['bin_bounds'] = (timestamp,)
        return df
    result = ShuffleSortFunctions.split_partitions_using_pivots_for_sort(df, columns_info, ascending, **kwargs)
    for i, pivot in enumerate(columns_info[0].pivots):
        add_attr(result[i], pivot - pandas.Timedelta(1, unit='ns'))
        if i + 1 <= len(result):
            add_attr(result[i + 1], pivot + pandas.Timedelta(1, unit='ns'))
    return result