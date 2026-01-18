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
def latex_visit_todo_node(self: LaTeXTranslator, node: todo_node) -> None:
    if self.config.todo_include_todos:
        self.body.append('\n\\begin{sphinxadmonition}{note}{')
        self.body.append(self.hypertarget_to(node))
        title_node = cast(nodes.title, node[0])
        title = texescape.escape(title_node.astext(), self.config.latex_engine)
        self.body.append('%s:}' % title)
        node.pop(0)
    else:
        raise nodes.SkipNode