import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
def find_name(self, nestedName: ASTNestedName, templateDecls: List[Any], typ: str, templateShorthand: bool, matchSelf: bool, recurseInAnon: bool, searchInSiblings: bool) -> Tuple[List['Symbol'], str]:
    if Symbol.debug_lookup:
        Symbol.debug_indent += 1
        Symbol.debug_print('find_name:')
        Symbol.debug_indent += 1
        Symbol.debug_print('self:')
        print(self.to_string(Symbol.debug_indent + 1), end='')
        Symbol.debug_print('nestedName:       ', nestedName)
        Symbol.debug_print('templateDecls:    ', templateDecls)
        Symbol.debug_print('typ:              ', typ)
        Symbol.debug_print('templateShorthand:', templateShorthand)
        Symbol.debug_print('matchSelf:        ', matchSelf)
        Symbol.debug_print('recurseInAnon:    ', recurseInAnon)
        Symbol.debug_print('searchInSiblings: ', searchInSiblings)

    class QualifiedSymbolIsTemplateParam(Exception):
        pass

    def onMissingQualifiedSymbol(parentSymbol: 'Symbol', identOrOp: Union[ASTIdentifier, ASTOperator], templateParams: Any, templateArgs: ASTTemplateArgs) -> 'Symbol':
        if parentSymbol.declaration is not None:
            if parentSymbol.declaration.objectType == 'templateParam':
                raise QualifiedSymbolIsTemplateParam()
        return None
    try:
        lookupResult = self._symbol_lookup(nestedName, templateDecls, onMissingQualifiedSymbol, strictTemplateParamArgLists=False, ancestorLookupType=typ, templateShorthand=templateShorthand, matchSelf=matchSelf, recurseInAnon=recurseInAnon, correctPrimaryTemplateArgs=False, searchInSiblings=searchInSiblings)
    except QualifiedSymbolIsTemplateParam:
        return (None, 'templateParamInQualified')
    if lookupResult is None:
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2
        return (None, None)
    res = list(lookupResult.symbols)
    if len(res) != 0:
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2
        return (res, None)
    if lookupResult.parentSymbol.declaration is not None:
        if lookupResult.parentSymbol.declaration.objectType == 'templateParam':
            return (None, 'templateParamInQualified')
    symbol = lookupResult.parentSymbol._find_first_named_symbol(lookupResult.identOrOp, None, None, templateShorthand=templateShorthand, matchSelf=matchSelf, recurseInAnon=recurseInAnon, correctPrimaryTemplateArgs=False)
    if Symbol.debug_lookup:
        Symbol.debug_indent -= 2
    if symbol is not None:
        return ([symbol], None)
    else:
        return (None, None)