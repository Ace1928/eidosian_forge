from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
@classmethod
def get_attribute_by_name(cls, el: bs4.Tag, name: str, default: str | Sequence[str] | None=None) -> str | Sequence[str] | None:
    """Get attribute by name."""
    value = default
    if el._is_xml:
        try:
            value = cls.normalize_value(el.attrs[name])
        except KeyError:
            pass
    else:
        for k, v in el.attrs.items():
            if util.lower(k) == name:
                value = cls.normalize_value(v)
                break
    return value