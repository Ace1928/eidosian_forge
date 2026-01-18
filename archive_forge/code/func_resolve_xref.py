import copy
from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List, NamedTuple, Optional,
from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst.states import Inliner
from sphinx.addnodes import pending_xref
from sphinx.errors import SphinxError
from sphinx.locale import _
from sphinx.roles import XRefRole
from sphinx.util.typing import RoleFunction
def resolve_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
    """Resolve the pending_xref *node* with the given *typ* and *target*.

        This method should return a new node, to replace the xref node,
        containing the *contnode* which is the markup content of the
        cross-reference.

        If no resolution can be found, None can be returned; the xref node will
        then given to the :event:`missing-reference` event, and if that yields no
        resolution, replaced by *contnode*.

        The method can also raise :exc:`sphinx.environment.NoUri` to suppress
        the :event:`missing-reference` event being emitted.
        """
    pass