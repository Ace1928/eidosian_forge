from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
from docutils import nodes
from docutils.nodes import Element, Node, make_id, system_message
from sphinx.addnodes import pending_xref
from sphinx.domains import Domain
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.nodes import make_refnode
def has_equations(self, docname: str=None) -> bool:
    if docname:
        return self.data['has_equations'].get(docname, False)
    else:
        return any(self.data['has_equations'].values())