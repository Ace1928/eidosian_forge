from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
def build_element(self, m: re.Match[str], builder: str, tags: str, index: int) -> etree.Element:
    """Element builder."""
    if builder == 'double2':
        return self.build_double2(m, tags, index)
    elif builder == 'double':
        return self.build_double(m, tags, index)
    else:
        return self.build_single(m, tags, index)