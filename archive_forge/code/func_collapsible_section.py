from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def collapsible_section(name, inline_details='', details='', n_items=None, enabled=True, collapsed=False) -> str:
    data_id = 'section-' + str(uuid.uuid4())
    has_items = n_items is not None and n_items
    n_items_span = '' if n_items is None else f' <span>({n_items})</span>'
    enabled = '' if enabled and has_items else 'disabled'
    collapsed = '' if collapsed or not has_items else 'checked'
    tip = " title='Expand/collapse section'" if enabled else ''
    return f"<input id='{data_id}' class='xr-section-summary-in' type='checkbox' {enabled} {collapsed}><label for='{data_id}' class='xr-section-summary' {tip}>{name}:{n_items_span}</label><div class='xr-section-inline-details'>{inline_details}</div><div class='xr-section-details'>{details}</div>"