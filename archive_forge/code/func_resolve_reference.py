from typing import Any, Dict, List, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.domains import Domain
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.util import logging, texescape
from sphinx.util.docutils import SphinxDirective, new_document
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
def resolve_reference(self, todo: todo_node, docname: str) -> None:
    """Resolve references in the todo content."""
    for node in todo.findall(addnodes.pending_xref):
        if 'refdoc' in node:
            node['refdoc'] = docname
    self.document += todo
    self.env.resolve_references(self.document, docname, self.builder)
    self.document.remove(todo)