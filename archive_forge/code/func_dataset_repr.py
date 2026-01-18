from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def dataset_repr(ds) -> str:
    obj_type = f'xarray.{type(ds).__name__}'
    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]
    sections = [dim_section(ds), coord_section(ds.coords), datavar_section(ds.data_vars), index_section(_get_indexes_dict(ds.xindexes)), attr_section(ds.attrs)]
    return _obj_repr(ds, header_components, sections)