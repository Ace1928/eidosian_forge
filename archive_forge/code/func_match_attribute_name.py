from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_attribute_name(self, el: bs4.Tag, attr: str, prefix: str | None) -> str | Sequence[str] | None:
    """Match attribute name and return value if it exists."""
    value = None
    if self.supports_namespaces():
        value = None
        if prefix:
            ns = self.namespaces.get(prefix)
            if ns is None and prefix != '*':
                return None
        else:
            ns = None
        for k, v in self.iter_attributes(el):
            namespace, name = self.split_namespace(el, k)
            if ns is None:
                if self.is_xml and attr == k or (not self.is_xml and util.lower(attr) == util.lower(k)):
                    value = v
                    break
                continue
            if namespace is None or (ns != namespace and prefix != '*'):
                continue
            if util.lower(attr) != util.lower(name) if not self.is_xml else attr != name:
                continue
            value = v
            break
    else:
        for k, v in self.iter_attributes(el):
            if util.lower(attr) != util.lower(k):
                continue
            value = v
            break
    return value