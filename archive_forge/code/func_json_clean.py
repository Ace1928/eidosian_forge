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
def json_clean(obj: Any) -> Any:
    atomic_ok = (str, type(None))
    container_to_list = (tuple, set, types.GeneratorType)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        if math.isnan(obj) or math.isinf(obj):
            return repr(obj)
        return float(obj)
    if isinstance(obj, atomic_ok):
        return obj
    if isinstance(obj, bytes):
        return b2a_base64(obj, newline=False).decode('ascii')
    if isinstance(obj, container_to_list) or (hasattr(obj, '__iter__') and hasattr(obj, next_attr_name)):
        obj = list(obj)
    if isinstance(obj, list):
        return [json_clean(x) for x in obj]
    if isinstance(obj, dict):
        nkeys = len(obj)
        nkeys_collapsed = len(set(map(str, obj)))
        if nkeys != nkeys_collapsed:
            msg = 'dict cannot be safely converted to JSON: key collision would lead to dropped values'
            raise ValueError(msg)
        out = {}
        for k, v in obj.items():
            out[str(k)] = json_clean(v)
        return out
    if isinstance(obj, datetime):
        return obj.strftime(ISO8601)
    raise ValueError("Can't clean for JSON: %r" % obj)