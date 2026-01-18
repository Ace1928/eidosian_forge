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
def build_toc_div(self, toc_list: list) -> etree.Element:
    """ Return a string div given a toc list. """
    div = etree.Element('div')
    div.attrib['class'] = self.toc_class
    if self.title:
        header = etree.SubElement(div, 'span')
        if self.title_class:
            header.attrib['class'] = self.title_class
        header.text = self.title

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
    build_etree_ul(toc_list, div)
    if 'prettify' in self.md.treeprocessors:
        self.md.treeprocessors['prettify'].run(div)
    return div