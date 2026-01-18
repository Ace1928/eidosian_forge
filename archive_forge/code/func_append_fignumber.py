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
def append_fignumber(figtype: str, figure_id: str) -> None:
    if self.builder.name == 'singlehtml':
        key = '%s/%s' % (self.docnames[-1], figtype)
    else:
        key = figtype
    if figure_id in self.builder.fignumbers.get(key, {}):
        self.body.append('<span class="caption-number">')
        prefix = self.config.numfig_format.get(figtype)
        if prefix is None:
            msg = __('numfig_format is not defined for %s') % figtype
            logger.warning(msg)
        else:
            numbers = self.builder.fignumbers[key][figure_id]
            self.body.append(prefix % '.'.join(map(str, numbers)) + ' ')
            self.body.append('</span>')