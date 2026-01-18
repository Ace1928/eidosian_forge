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
def get_prev_node(node: Node) -> Optional[Node]:
    pos = node.parent.index(node)
    if pos > 0:
        return node.parent[pos - 1]
    else:
        return None