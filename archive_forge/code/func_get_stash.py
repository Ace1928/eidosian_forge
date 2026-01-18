from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
def get_stash(m: re.Match[str]) -> str:
    id = m.group(1)
    value = stash.get(id)
    if value is not None:
        try:
            return self.md.serializer(value)
        except Exception:
            return '\\%s' % value