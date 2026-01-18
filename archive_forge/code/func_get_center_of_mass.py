from __future__ import annotations
import datetime
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.tslibs import Timedelta
import pandas._libs.window.aggregations as window_aggregations
from pandas.util._decorators import doc
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core import common
from pandas.core.arrays.datetimelike import dtype_to_unit
from pandas.core.indexers.objects import (
from pandas.core.util.numba_ import (
from pandas.core.window.common import zsqrt
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.window.online import (
from pandas.core.window.rolling import (
def get_center_of_mass(comass: float | None, span: float | None, halflife: float | None, alpha: float | None) -> float:
    valid_count = common.count_not_none(comass, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError('comass, span, halflife, and alpha are mutually exclusive')
    if comass is not None:
        if comass < 0:
            raise ValueError('comass must satisfy: comass >= 0')
    elif span is not None:
        if span < 1:
            raise ValueError('span must satisfy: span >= 1')
        comass = (span - 1) / 2
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError('halflife must satisfy: halflife > 0')
        decay = 1 - np.exp(np.log(0.5) / halflife)
        comass = 1 / decay - 1
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError('alpha must satisfy: 0 < alpha <= 1')
        comass = (1 - alpha) / alpha
    else:
        raise ValueError('Must pass one of comass, span, halflife, or alpha')
    return float(comass)