from typing import Any, Dict, List, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.transforms import SphinxTransform
def invisible_visit(self, node: Node) -> None:
    """Invisible nodes should be ignored."""
    pass