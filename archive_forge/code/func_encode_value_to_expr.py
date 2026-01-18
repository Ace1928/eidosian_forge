from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
from triad import Schema
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP
from triad.utils.schema import quote_name
def encode_value_to_expr(value: Any) -> str:
    if isinstance(value, list):
        return '[' + ', '.join((encode_value_to_expr(x) for x in value)) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join((encode_value_to_expr(k) + ': ' + encode_value_to_expr(v) for k, v in value.items())) + '}'
    if pd.isna(value):
        return 'NULL'
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return 'E' + repr(value)
    if isinstance(value, bytes):
        return repr(value)[1:] + '::BLOB'
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, datetime):
        return "TIMESTAMP '" + value.strftime('%Y-%m-%d %H:%M:%S') + "'"
    if isinstance(value, date):
        return "DATE '" + value.strftime('%Y-%m-%d') + "'"
    raise NotImplementedError(value)