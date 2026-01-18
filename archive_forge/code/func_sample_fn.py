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
def sample_fn(self, partition: pandas.DataFrame) -> pandas.DataFrame:
    if self.level is not None:
        partition = self._index_to_df_zero_copy(partition, self.level)
    else:
        partition = partition[self.columns]
    return self.pick_samples_for_quantiles(partition, self.ideal_num_new_partitions, self.frame_len)