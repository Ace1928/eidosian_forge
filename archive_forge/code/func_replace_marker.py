from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue, AMP_SUBSTITUTE, deprecated, HTML_PLACEHOLDER_RE, AtomicString
from ..treeprocessors import UnescapeTreeprocessor
from ..serializers import RE_AMP
import re
import html
import unicodedata
from copy import deepcopy
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any, Iterator, MutableSet
def replace_marker(self, root: etree.Element, elem: etree.Element) -> None:
    """ Replace marker with elem. """
    for p, c in self.iterparent(root):
        text = ''.join(c.itertext()).strip()
        if not text:
            continue
        if c.text and c.text.strip() == self.marker and (len(c) == 0):
            for i in range(len(p)):
                if p[i] == c:
                    p[i] = elem
                    break