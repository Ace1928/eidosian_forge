from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def summarize_attrs(attrs) -> str:
    attrs_dl = ''.join((f'<dt><span>{escape(str(k))} :</span></dt><dd>{escape(str(v))}</dd>' for k, v in attrs.items()))
    return f"<dl class='xr-attrs'>{attrs_dl}</dl>"