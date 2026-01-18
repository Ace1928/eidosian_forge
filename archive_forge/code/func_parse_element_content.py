from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..preprocessors import Preprocessor
from ..postprocessors import RawHtmlPostprocessor
from .. import util
from ..htmlparser import HTMLExtractor, blank_line_re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Literal, Mapping
def parse_element_content(self, element: etree.Element) -> None:
    """
        Recursively parse the text content of an `etree` Element as Markdown.

        Any block level elements generated from the Markdown will be inserted as children of the element in place
        of the text content. All `markdown` attributes are removed. For any elements in which Markdown parsing has
        been disabled, the text content of it and its children are wrapped in an `AtomicString`.
        """
    md_attr = element.attrib.pop('markdown', 'off')
    if md_attr == 'block':
        for child in list(element):
            self.parse_element_content(child)
        tails = []
        for pos, child in enumerate(element):
            if child.tail:
                block = child.tail.rstrip('\n')
                child.tail = ''
                dummy = etree.Element('div')
                self.parser.parseBlocks(dummy, block.split('\n\n'))
                children = list(dummy)
                children.reverse()
                tails.append((pos + 1, children))
        tails.reverse()
        for pos, tail in tails:
            for item in tail:
                element.insert(pos, item)
        if element.text:
            block = element.text.rstrip('\n')
            element.text = ''
            dummy = etree.Element('div')
            self.parser.parseBlocks(dummy, block.split('\n\n'))
            children = list(dummy)
            children.reverse()
            for child in children:
                element.insert(0, child)
    elif md_attr == 'span':
        for child in list(element):
            self.parse_element_content(child)
    else:
        if element.text is None:
            element.text = ''
        element.text = util.AtomicString(element.text)
        for child in list(element):
            self.parse_element_content(child)
            if child.tail:
                child.tail = util.AtomicString(child.tail)