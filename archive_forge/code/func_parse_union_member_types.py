from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_union_member_types(self) -> List[NamedTypeNode]:
    """UnionMemberTypes"""
    return self.delimited_many(TokenKind.PIPE, self.parse_named_type) if self.expect_optional_token(TokenKind.EQUALS) else []