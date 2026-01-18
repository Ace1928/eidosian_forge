from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def summarize_vars(variables) -> str:
    vars_li = ''.join((f"<li class='xr-var-item'>{summarize_variable(k, v)}</li>" for k, v in variables.items()))
    return f"<ul class='xr-var-list'>{vars_li}</ul>"