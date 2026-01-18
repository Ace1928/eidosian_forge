import os
import posixpath
import re
import urllib.parse
import warnings
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node, Text
from docutils.writers.html4css1 import HTMLTranslator as BaseTranslator
from docutils.writers.html4css1 import Writer
from sphinx import addnodes
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.images import get_image_size
def multiply_length(length: str, scale: int) -> str:
    """Multiply *length* (width or height) by *scale*."""
    matched = re.match('^(\\d*\\.?\\d*)\\s*(\\S*)$', length)
    if not matched:
        return length
    elif scale == 100:
        return length
    else:
        amount, unit = matched.groups()
        result = float(amount) * scale / 100
        return '%s%s' % (int(result), unit)