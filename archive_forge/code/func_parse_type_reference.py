from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_type_reference(self) -> TypeNode:
    """Type: NamedType or ListType or NonNullType"""
    start = self._lexer.token
    type_: TypeNode
    if self.expect_optional_token(TokenKind.BRACKET_L):
        inner_type = self.parse_type_reference()
        self.expect_token(TokenKind.BRACKET_R)
        type_ = ListTypeNode(type=inner_type, loc=self.loc(start))
    else:
        type_ = self.parse_named_type()
    if self.expect_optional_token(TokenKind.BANG):
        return NonNullTypeNode(type=type_, loc=self.loc(start))
    return type_