from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas._typing import (
from pandas.util._validators import validate_percentile
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.reshape.concat import concat
from pandas.io.formats.format import format_percentiles
def describe_numeric_1d(series: Series, percentiles: Sequence[float]) -> Series:
    """Describe series containing numerical data.

    Parameters
    ----------
    series : Series
        Series to be described.
    percentiles : list-like of numbers
        The percentiles to include in the output.
    """
    from pandas import Series
    formatted_percentiles = format_percentiles(percentiles)
    stat_index = ['count', 'mean', 'std', 'min'] + formatted_percentiles + ['max']
    d = [series.count(), series.mean(), series.std(), series.min()] + series.quantile(percentiles).tolist() + [series.max()]
    dtype: DtypeObj | None
    if isinstance(series.dtype, ExtensionDtype):
        if isinstance(series.dtype, ArrowDtype):
            if series.dtype.kind == 'm':
                dtype = None
            else:
                import pyarrow as pa
                dtype = ArrowDtype(pa.float64())
        else:
            dtype = Float64Dtype()
    elif series.dtype.kind in 'iufb':
        dtype = np.dtype('float')
    else:
        dtype = None
    return Series(d, index=stat_index, name=series.name, dtype=dtype)