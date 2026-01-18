from __future__ import annotations
import ctypes
import re
from typing import Any
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SettingWithCopyError
import pandas as pd
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (
def parse_datetime_format_str(format_str, data) -> pd.Series | np.ndarray:
    """Parse datetime `format_str` to interpret the `data`."""
    timestamp_meta = re.match('ts([smun]):(.*)', format_str)
    if timestamp_meta:
        unit, tz = (timestamp_meta.group(1), timestamp_meta.group(2))
        if unit != 's':
            unit += 's'
        data = data.astype(f'datetime64[{unit}]')
        if tz != '':
            data = pd.Series(data).dt.tz_localize('UTC').dt.tz_convert(tz)
        return data
    date_meta = re.match('td([Dm])', format_str)
    if date_meta:
        unit = date_meta.group(1)
        if unit == 'D':
            data = (data.astype(np.uint64) * (24 * 60 * 60)).astype('datetime64[s]')
        elif unit == 'm':
            data = data.astype('datetime64[ms]')
        else:
            raise NotImplementedError(f'Date unit is not supported: {unit}')
        return data
    raise NotImplementedError(f'DateTime kind is not supported: {format_str}')