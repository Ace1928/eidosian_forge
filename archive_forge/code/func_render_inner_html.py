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
def render_inner_html(el: etree.Element, md: Markdown) -> str:
    """ Fully render inner html of an `etree` element as a string. """
    text = unescape(md.serializer(el))
    start = text.index('>') + 1
    end = text.rindex('<')
    text = text[start:end].strip()
    return run_postprocessors(text, md)