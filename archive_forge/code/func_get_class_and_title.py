from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
from typing import TYPE_CHECKING
def get_class_and_title(self, match: re.Match[str]) -> tuple[str, str | None]:
    klass, title = (match.group(1).lower(), match.group(2))
    klass = self.RE_SPACES.sub(' ', klass)
    if title is None:
        title = klass.split(' ', 1)[0].capitalize()
    elif title == '':
        title = None
    return (klass, title)