import re
import sys
from copy import copy
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional,
from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.locale import _, __
from sphinx.roles import EmphasizedLiteral, XRefRole
from sphinx.util import docname_join, logging, ws_re
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import clean_astext, make_id, make_refnode
from sphinx.util.typing import OptionSpec, RoleFunction
def get_numfig_title(self, node: Node) -> Optional[str]:
    """Get the title of enumerable nodes to refer them using its title"""
    if self.is_enumerable_node(node):
        elem = cast(Element, node)
        _, title_getter = self.enumerable_nodes.get(elem.__class__, (None, None))
        if title_getter:
            return title_getter(elem)
        else:
            for subnode in elem:
                if isinstance(subnode, (nodes.caption, nodes.title)):
                    return clean_astext(subnode)
    return None