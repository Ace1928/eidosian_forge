import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type
import docutils.parsers.rst.directives
import docutils.parsers.rst.roles
import docutils.parsers.rst.states
from docutils import nodes, utils
from docutils.nodes import Element, Node, TextElement, system_message
from sphinx import addnodes
from sphinx.locale import _, __
from sphinx.util import ws_re
from sphinx.util.docutils import ReferenceRole, SphinxRole
from sphinx.util.typing import RoleFunction
def create_non_xref_node(self) -> Tuple[List[Node], List[system_message]]:
    text = utils.unescape(self.text[1:])
    if self.fix_parens:
        self.has_explicit_title = False
        text, target = self.update_title_and_target(text, '')
    node = self.innernodeclass(self.rawtext, text, classes=self.classes)
    return self.result_nodes(self.inliner.document, self.env, node, is_ref=False)