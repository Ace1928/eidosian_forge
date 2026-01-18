from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs import lib
from pandas._libs.algos import unique_deltas
from pandas._libs.tslibs import (
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.parsing import get_rule_month
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import is_numeric_dtype
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.algorithms import unique
def is_superperiod(source, target) -> bool:
    """
    Returns True if upsampling is possible between source and target
    frequencies

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from
    target : str or DateOffset
        Frequency converting to

    Returns
    -------
    bool
    """
    if target is None or source is None:
        return False
    source = _maybe_coerce_freq(source)
    target = _maybe_coerce_freq(target)
    if _is_annual(source):
        if _is_annual(target):
            return get_rule_month(source) == get_rule_month(target)
        if _is_quarterly(target):
            smonth = get_rule_month(source)
            tmonth = get_rule_month(target)
            return _quarter_months_conform(smonth, tmonth)
        return target in {'D', 'C', 'B', 'M', 'h', 'min', 's', 'ms', 'us', 'ns'}
    elif _is_quarterly(source):
        return target in {'D', 'C', 'B', 'M', 'h', 'min', 's', 'ms', 'us', 'ns'}
    elif _is_monthly(source):
        return target in {'D', 'C', 'B', 'h', 'min', 's', 'ms', 'us', 'ns'}
    elif _is_weekly(source):
        return target in {source, 'D', 'C', 'B', 'h', 'min', 's', 'ms', 'us', 'ns'}
    elif source == 'B':
        return target in {'D', 'C', 'B', 'h', 'min', 's', 'ms', 'us', 'ns'}
    elif source == 'C':
        return target in {'D', 'C', 'B', 'h', 'min', 's', 'ms', 'us', 'ns'}
    elif source == 'D':
        return target in {'D', 'C', 'B', 'h', 'min', 's', 'ms', 'us', 'ns'}
    elif source == 'h':
        return target in {'h', 'min', 's', 'ms', 'us', 'ns'}
    elif source == 'min':
        return target in {'min', 's', 'ms', 'us', 'ns'}
    elif source == 's':
        return target in {'s', 'ms', 'us', 'ns'}
    elif source == 'ms':
        return target in {'ms', 'us', 'ns'}
    elif source == 'us':
        return target in {'us', 'ns'}
    elif source == 'ns':
        return target in {'ns'}
    else:
        return False