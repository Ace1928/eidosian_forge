from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..treeprocessors import Treeprocessor
from ..postprocessors import Postprocessor
from .. import util
from collections import OrderedDict
import re
import copy
import xml.etree.ElementTree as etree
def makeFootnotesDiv(self, root: etree.Element) -> etree.Element | None:
    """ Return `div` of footnotes as `etree` Element. """
    if not list(self.footnotes.keys()):
        return None
    div = etree.Element('div')
    div.set('class', 'footnote')
    etree.SubElement(div, 'hr')
    ol = etree.SubElement(div, 'ol')
    surrogate_parent = etree.Element('div')
    backlink_title = self.getConfig('BACKLINK_TITLE').replace('%d', '{}')
    for index, id in enumerate(self.footnotes.keys(), start=1):
        li = etree.SubElement(ol, 'li')
        li.set('id', self.makeFootnoteId(id))
        self.parser.parseChunk(surrogate_parent, self.footnotes[id])
        for el in list(surrogate_parent):
            li.append(el)
            surrogate_parent.remove(el)
        backlink = etree.Element('a')
        backlink.set('href', '#' + self.makeFootnoteRefId(id))
        backlink.set('class', 'footnote-backref')
        backlink.set('title', backlink_title.format(index))
        backlink.text = FN_BACKLINK_TEXT
        if len(li):
            node = li[-1]
            if node.tag == 'p':
                node.text = node.text + NBSP_PLACEHOLDER
                node.append(backlink)
            else:
                p = etree.SubElement(li, 'p')
                p.append(backlink)
    return div