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
def date_default(obj: Any) -> Any:
    """DEPRECATED: Use jupyter_client.jsonutil.json_default"""
    warnings.warn('date_default is deprecated since jupyter_client 7.0.0. Use jupyter_client.jsonutil.json_default.', stacklevel=2)
    return json_default(obj)