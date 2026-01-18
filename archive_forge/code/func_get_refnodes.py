import html
import os
import re
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.utils import smartquotes
from sphinx import addnodes
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.locale import __
from sphinx.util import logging, status_iterator
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import copyfile, ensuredir
def get_refnodes(self, doctree: Node, result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect section titles, their depth in the toc and the refuri."""
    if isinstance(doctree, nodes.reference) and doctree.get('refuri'):
        refuri = doctree['refuri']
        if refuri.startswith('http://') or refuri.startswith('https://') or refuri.startswith('irc:') or refuri.startswith('mailto:'):
            return result
        classes = doctree.parent.attributes['classes']
        for level in range(8, 0, -1):
            if self.toctree_template % level in classes:
                result.append({'level': level, 'refuri': html.escape(refuri), 'text': ssp(html.escape(doctree.astext()))})
                break
    elif isinstance(doctree, nodes.Element):
        for elem in doctree:
            result = self.get_refnodes(elem, result)
    return result