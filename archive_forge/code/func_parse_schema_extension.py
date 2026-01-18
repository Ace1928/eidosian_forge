from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_schema_extension(self) -> SchemaExtensionNode:
    """SchemaExtension"""
    start = self._lexer.token
    self.expect_keyword('extend')
    self.expect_keyword('schema')
    directives = self.parse_const_directives()
    operation_types = self.optional_many(TokenKind.BRACE_L, self.parse_operation_type_definition, TokenKind.BRACE_R)
    if not directives and (not operation_types):
        raise self.unexpected()
    return SchemaExtensionNode(directives=directives, operation_types=operation_types, loc=self.loc(start))