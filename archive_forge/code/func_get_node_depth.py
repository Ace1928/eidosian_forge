from typing import Any, Dict, cast
from docutils import nodes
from docutils.nodes import Node
import sphinx
from sphinx.application import Sphinx
from sphinx.domains.std import StandardDomain
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.nodes import clean_astext
def get_node_depth(node: Node) -> int:
    i = 0
    cur_node = node
    while cur_node.parent != node.document:
        cur_node = cur_node.parent
        i += 1
    return i