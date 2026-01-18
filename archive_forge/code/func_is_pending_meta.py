import re
import unicodedata
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple, Type,
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging
def is_pending_meta(node: Node) -> bool:
    if isinstance(node, nodes.pending) and isinstance(node.details.get('nodes', [None])[0], addnodes.meta):
        return True
    else:
        return False