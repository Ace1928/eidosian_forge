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
def build_navpoints(self, nodes: List[Dict[str, Any]]) -> List[NavPoint]:
    """Create the toc navigation structure.

        Subelements of a node are nested inside the navpoint.  For nested nodes
        the parent node is reinserted in the subnav.
        """
    navstack: List[NavPoint] = []
    navstack.append(NavPoint('dummy', 0, '', '', []))
    level = 0
    lastnode = None
    for node in nodes:
        if not node['text']:
            continue
        file = node['refuri'].split('#')[0]
        if file in self.ignored_files:
            continue
        if node['level'] > self.config.epub_tocdepth:
            continue
        if node['level'] == level:
            navpoint = self.new_navpoint(node, level)
            navstack.pop()
            navstack[-1].children.append(navpoint)
            navstack.append(navpoint)
        elif node['level'] == level + 1:
            level += 1
            if lastnode and self.config.epub_tocdup:
                navstack[-1].children.append(self.new_navpoint(lastnode, level, False))
            navpoint = self.new_navpoint(node, level)
            navstack[-1].children.append(navpoint)
            navstack.append(navpoint)
        elif node['level'] < level:
            while node['level'] < len(navstack):
                navstack.pop()
            level = node['level']
            navpoint = self.new_navpoint(node, level)
            navstack[-1].children.append(navpoint)
            navstack.append(navpoint)
        else:
            raise
        lastnode = node
    return navstack[0].children