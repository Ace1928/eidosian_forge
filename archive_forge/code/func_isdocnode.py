from __future__ import annotations
import html
import os
import posixpath
import re
from collections.abc import Iterable
from os import path
from typing import Any, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import canon_path, make_filename
from sphinx.util.template import SphinxRenderer
def isdocnode(self, node: Node) -> bool:
    if not isinstance(node, nodes.list_item):
        return False
    if len(node.children) != 2:
        return False
    if not isinstance(node[0], addnodes.compact_paragraph):
        return False
    if not isinstance(node[0][0], nodes.reference):
        return False
    if not isinstance(node[1], nodes.bullet_list):
        return False
    return True