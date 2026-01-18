import math
import numbers
import re
import types
import warnings
from binascii import b2a_base64
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Optional, Union
from dateutil.parser import parse as _dateutil_parse
from dateutil.tz import tzlocal
def json_default(obj: Any) -> Any:
    """default function for packing objects in JSON."""
    if isinstance(obj, datetime):
        obj = _ensure_tzinfo(obj)
        return obj.isoformat().replace('+00:00', 'Z')
    if isinstance(obj, bytes):
        return b2a_base64(obj, newline=False).decode('ascii')
    if isinstance(obj, Iterable):
        return list(obj)
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return float(obj)
    raise TypeError('%r is not JSON serializable' % obj)