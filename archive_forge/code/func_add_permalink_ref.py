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
def add_permalink_ref(self, node: Element, title: str) -> None:
    if node['ids'] and self.config.html_permalinks and self.builder.add_permalinks:
        format = '<a class="headerlink" href="#%s" title="%s">%s</a>'
        self.body.append(format % (node['ids'][0], title, self.config.html_permalinks_icon))