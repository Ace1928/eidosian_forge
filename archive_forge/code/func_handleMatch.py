from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..util import AtomicString
import re
import xml.etree.ElementTree as etree
def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
    abbr = etree.Element('abbr')
    abbr.text = AtomicString(m.group('abbr'))
    abbr.set('title', self.title)
    return (abbr, m.start(0), m.end(0))