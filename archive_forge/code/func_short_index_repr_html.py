from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def short_index_repr_html(index) -> str:
    if hasattr(index, '_repr_html_'):
        return index._repr_html_()
    return f'<pre>{escape(repr(index))}</pre>'