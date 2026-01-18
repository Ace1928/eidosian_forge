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
def create_todo_reference(self, todo: todo_node, docname: str) -> nodes.paragraph:
    if self.config.todo_link_only:
        description = _('<<original entry>>')
    else:
        description = _('(The <<original entry>> is located in %s, line %d.)') % (todo.source, todo.line)
    prefix = description[:description.find('<<')]
    suffix = description[description.find('>>') + 2:]
    para = nodes.paragraph(classes=['todo-source'])
    para += nodes.Text(prefix)
    linktext = nodes.emphasis(_('original entry'), _('original entry'))
    reference = nodes.reference('', '', linktext, internal=True)
    try:
        reference['refuri'] = self.builder.get_relative_uri(docname, todo['docname'])
        reference['refuri'] += '#' + todo['ids'][0]
    except NoUri:
        pass
    para += reference
    para += nodes.Text(suffix)
    return para