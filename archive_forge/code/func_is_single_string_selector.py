from __future__ import annotations
import logging # isort:skip
from typing import (
from ..model import Model
def is_single_string_selector(selector: SelectorType, field: str) -> bool:
    """ Whether a selector is a simple single field, e.g. ``{name: "foo"}``

    Args:
        selector (JSON-like) : query selector
        field (str) : field name to check for

    Returns
        bool

    """
    return len(selector) == 1 and field in selector and isinstance(selector[field], str)