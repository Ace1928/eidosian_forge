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
def build_etree_ul(toc_list: list, parent: etree.Element) -> etree.Element:
    ul = etree.SubElement(parent, 'ul')
    for item in toc_list:
        li = etree.SubElement(ul, 'li')
        link = etree.SubElement(li, 'a')
        link.text = item.get('name', '')
        link.attrib['href'] = '#' + item.get('id', '')
        if item['children']:
            build_etree_ul(item['children'], li)
    return ul