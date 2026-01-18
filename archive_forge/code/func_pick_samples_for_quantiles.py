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
def pick_samples_for_quantiles(df: pandas.DataFrame, num_partitions: int, length: int) -> pandas.DataFrame:
    return pandas.concat([df.min().to_frame().T, df.max().to_frame().T])