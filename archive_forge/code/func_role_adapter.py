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
def role_adapter(typ: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options: Dict={}, content: List[str]=[]) -> Tuple[List[Node], List[system_message]]:
    return self.roles[name](fullname, rawtext, text, lineno, inliner, options, content)