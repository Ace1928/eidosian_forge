import builtins
import inspect
import re
import sys
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref, pending_xref_condition
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, Index, IndexEntry, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import (find_pending_xref_condition, make_id, make_refnode,
from sphinx.util.typing import OptionSpec, TextlikeNode
def builtin_resolver(app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: Element) -> Element:
    """Do not emit nitpicky warnings for built-in types."""

    def istyping(s: str) -> bool:
        if s.startswith('typing.'):
            s = s.split('.', 1)[1]
        return s in typing.__all__
    if node.get('refdomain') != 'py':
        return None
    elif node.get('reftype') in ('class', 'obj') and node.get('reftarget') == 'None':
        return contnode
    elif node.get('reftype') in ('class', 'obj', 'exc'):
        reftarget = node.get('reftarget')
        if inspect.isclass(getattr(builtins, reftarget, None)):
            return contnode
        elif istyping(reftarget):
            return contnode
    return None