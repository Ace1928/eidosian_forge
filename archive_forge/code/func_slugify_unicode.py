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
def slugify_unicode(value: str, separator: str) -> str:
    """ Slugify a string, to make it URL friendly while preserving Unicode characters. """
    return slugify(value, separator, unicode=True)