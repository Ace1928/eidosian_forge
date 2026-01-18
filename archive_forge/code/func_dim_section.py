from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def dim_section(obj) -> str:
    dim_list = format_dims(obj.sizes, obj.xindexes.dims)
    return collapsible_section('Dimensions', inline_details=dim_list, enabled=False, collapsed=True)