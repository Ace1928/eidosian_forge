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
def pick_pivots_from_samples_for_sort(self, samples: np.ndarray, ideal_num_new_partitions: int, method: str='linear', key: Optional[Callable]=None) -> np.ndarray:
    if key is not None:
        raise NotImplementedError(key)
    max_value = samples.max()
    first, last = _get_timestamp_range_edges(samples.min(), max_value, self.resample_kwargs['freq'], unit=samples.dt.unit, closed=self.resample_kwargs['closed'], origin=self.resample_kwargs['origin'], offset=self.resample_kwargs['offset'])
    all_bins = pandas.date_range(start=first, end=last, freq=self.resample_kwargs['freq'], ambiguous=True, nonexistent='shift_forward', unit=samples.dt.unit)
    all_bins = self._adjust_bin_edges(all_bins, max_value, freq=self.resample_kwargs['freq'], closed=self.resample_kwargs['closed'])
    step = 1 / ideal_num_new_partitions
    bins = [all_bins[int(len(all_bins) * i * step)] for i in range(1, ideal_num_new_partitions)]
    return bins