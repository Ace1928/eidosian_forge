from __future__ import annotations
from typing import (
from pyarrow.interchange.column import (
import pyarrow as pa
import re
import pyarrow.compute as pc
from pyarrow.interchange.column import Dtype
def map_date_type(data_type):
    """Map column date type to pyarrow date type. """
    kind, bit_width, f_string, _ = data_type
    if kind == DtypeKind.DATETIME:
        unit, tz = parse_datetime_format_str(f_string)
        return pa.timestamp(unit, tz=tz)
    else:
        pa_dtype = _PYARROW_DTYPES.get(kind, {}).get(bit_width, None)
        if pa_dtype:
            return pa_dtype
        else:
            raise NotImplementedError(f'Conversion for {data_type} is not yet supported.')