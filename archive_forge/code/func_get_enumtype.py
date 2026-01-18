import re
import warnings
from collections import defaultdict
from os import path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, cast
from docutils import nodes, writers
from docutils.nodes import Element, Node, Text
from sphinx import addnodes, highlighting
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.domains import IndexEntry
from sphinx.domains.std import StandardDomain
from sphinx.errors import SphinxError
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging, split_into, texescape
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.nodes import clean_astext, get_prev_node
from sphinx.util.template import LaTeXRenderer
from sphinx.util.texescape import tex_replace_map
from sphinx.builders.latex.nodes import ( # NOQA isort:skip
def get_enumtype(node: Element) -> str:
    enumtype = node.get('enumtype', 'arabic')
    if 'alpha' in enumtype and 26 < node.get('start', 0) + len(node):
        enumtype = 'arabic'
    return enumtype