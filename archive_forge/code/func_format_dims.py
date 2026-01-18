from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def format_dims(dim_sizes, dims_with_index) -> str:
    if not dim_sizes:
        return ''
    dim_css_map = {dim: " class='xr-has-index'" if dim in dims_with_index else '' for dim in dim_sizes}
    dims_li = ''.join((f'<li><span{dim_css_map[dim]}>{escape(str(dim))}</span>: {size}</li>' for dim, size in dim_sizes.items()))
    return f"<ul class='xr-dim-list'>{dims_li}</ul>"