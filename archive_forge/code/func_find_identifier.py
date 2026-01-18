import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
def find_identifier(self, ident: ASTIdentifier, matchSelf: bool, recurseInAnon: bool, searchInSiblings: bool) -> 'Symbol':
    if Symbol.debug_lookup:
        Symbol.debug_indent += 1
        Symbol.debug_print('find_identifier:')
        Symbol.debug_indent += 1
        Symbol.debug_print('ident:           ', ident)
        Symbol.debug_print('matchSelf:       ', matchSelf)
        Symbol.debug_print('recurseInAnon:   ', recurseInAnon)
        Symbol.debug_print('searchInSiblings:', searchInSiblings)
        print(self.to_string(Symbol.debug_indent + 1), end='')
        Symbol.debug_indent -= 2
    current = self
    while current is not None:
        if Symbol.debug_lookup:
            Symbol.debug_indent += 2
            Symbol.debug_print('trying:')
            print(current.to_string(Symbol.debug_indent + 1), end='')
            Symbol.debug_indent -= 2
        if matchSelf and current.ident == ident:
            return current
        children = current.children_recurse_anon if recurseInAnon else current._children
        for s in children:
            if s.ident == ident:
                return s
        if not searchInSiblings:
            break
        current = current.siblingAbove
    return None