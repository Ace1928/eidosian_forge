from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_union_type_extension(self) -> UnionTypeExtensionNode:
    """UnionTypeExtension"""
    start = self._lexer.token
    self.expect_keyword('extend')
    self.expect_keyword('union')
    name = self.parse_name()
    directives = self.parse_const_directives()
    types = self.parse_union_member_types()
    if not (directives or types):
        raise self.unexpected()
    return UnionTypeExtensionNode(name=name, directives=directives, types=types, loc=self.loc(start))