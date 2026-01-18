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
def add_object_type(self, name: str, objtype: ObjType) -> None:
    """Add an object type."""
    self.object_types[name] = objtype
    if objtype.roles:
        self._type2role[name] = objtype.roles[0]
    else:
        self._type2role[name] = ''
    for role in objtype.roles:
        self._role2type.setdefault(role, []).append(name)