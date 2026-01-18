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
def find_declaration(self, nestedName: ASTNestedName, typ: str, matchSelf: bool, recurseInAnon: bool) -> 'Symbol':
    if Symbol.debug_lookup:
        Symbol.debug_indent += 1
        Symbol.debug_print('find_declaration:')

    def onMissingQualifiedSymbol(parentSymbol: 'Symbol', ident: ASTIdentifier) -> 'Symbol':
        return None
    lookupResult = self._symbol_lookup(nestedName, onMissingQualifiedSymbol, ancestorLookupType=typ, matchSelf=matchSelf, recurseInAnon=recurseInAnon, searchInSiblings=False)
    if Symbol.debug_lookup:
        Symbol.debug_indent -= 1
    if lookupResult is None:
        return None
    symbols = list(lookupResult.symbols)
    if len(symbols) == 0:
        return None
    return symbols[0]