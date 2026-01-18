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
def add_permalink(self, c: etree.Element, elem_id: str) -> None:
    permalink = etree.Element('a')
    permalink.text = '%spara;' % AMP_SUBSTITUTE if self.use_permalinks is True else self.use_permalinks
    permalink.attrib['href'] = '#' + elem_id
    permalink.attrib['class'] = self.permalink_class
    if self.permalink_title:
        permalink.attrib['title'] = self.permalink_title
    if self.permalink_leading:
        permalink.tail = c.text
        c.text = ''
        c.insert(0, permalink)
    else:
        c.append(permalink)