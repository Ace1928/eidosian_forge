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
def note_equation(self, docname: str, labelid: str, location: Any=None) -> None:
    if labelid in self.equations:
        other = self.equations[labelid][0]
        logger.warning(__('duplicate label of equation %s, other instance in %s') % (labelid, other), location=location)
    self.equations[labelid] = (docname, self.env.new_serialno('eqno') + 1)