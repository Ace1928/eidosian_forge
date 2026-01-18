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
def new_navpoint(self, node: Dict[str, Any], level: int, incr: bool=True) -> NavPoint:
    """Create a new entry in the toc from the node at given level."""
    if incr:
        self.playorder += 1
    self.tocid += 1
    return NavPoint('navPoint%d' % self.tocid, self.playorder, node['text'], node['refuri'], [])