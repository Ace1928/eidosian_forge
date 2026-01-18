from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
def parse_sub_patterns(self, data: str, parent: etree.Element, last: etree.Element | None, idx: int) -> None:
    """
        Parses sub patterns.

        `data`: text to evaluate.

        `parent`: Parent to attach text and sub elements to.

        `last`: Last appended child to parent. Can also be None if parent has no children.

        `idx`: Current pattern index that was used to evaluate the parent.
        """
    offset = 0
    pos = 0
    length = len(data)
    while pos < length:
        if self.compiled_re.match(data, pos):
            matched = False
            for index, item in enumerate(self.PATTERNS):
                if index <= idx:
                    continue
                m = item.pattern.match(data, pos)
                if m:
                    text = data[offset:m.start(0)]
                    if text:
                        if last is not None:
                            last.tail = text
                        else:
                            parent.text = text
                    el = self.build_element(m, item.builder, item.tags, index)
                    parent.append(el)
                    last = el
                    offset = pos = m.end(0)
                    matched = True
            if not matched:
                pos += 1
        else:
            pos += 1
    text = data[offset:]
    if text:
        if last is not None:
            last.tail = text
        else:
            parent.text = text