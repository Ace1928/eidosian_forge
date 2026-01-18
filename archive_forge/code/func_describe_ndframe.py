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
def describe_ndframe(*, obj: NDFrameT, include: str | Sequence[str] | None, exclude: str | Sequence[str] | None, percentiles: Sequence[float] | np.ndarray | None) -> NDFrameT:
    """Describe series or dataframe.

    Called from pandas.core.generic.NDFrame.describe()

    Parameters
    ----------
    obj: DataFrame or Series
        Either dataframe or series to be described.
    include : 'all', list-like of dtypes or None (default), optional
        A white list of data types to include in the result. Ignored for ``Series``.
    exclude : list-like of dtypes or None (default), optional,
        A black list of data types to omit from the result. Ignored for ``Series``.
    percentiles : list-like of numbers, optional
        The percentiles to include in the output. All should fall between 0 and 1.
        The default is ``[.25, .5, .75]``, which returns the 25th, 50th, and
        75th percentiles.

    Returns
    -------
    Dataframe or series description.
    """
    percentiles = _refine_percentiles(percentiles)
    describer: NDFrameDescriberAbstract
    if obj.ndim == 1:
        describer = SeriesDescriber(obj=cast('Series', obj))
    else:
        describer = DataFrameDescriber(obj=cast('DataFrame', obj), include=include, exclude=exclude)
    result = describer.describe(percentiles=percentiles)
    return cast(NDFrameT, result)