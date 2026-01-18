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
def add_secnumber(self, node: Element) -> None:
    secnumber = self.get_secnumber(node)
    if secnumber:
        self.body.append('<span class="section-number">%s</span>' % ('.'.join(map(str, secnumber)) + self.secnumber_suffix))