from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_enum_type_extension(self) -> EnumTypeExtensionNode:
    """EnumTypeExtension"""
    start = self._lexer.token
    self.expect_keyword('extend')
    self.expect_keyword('enum')
    name = self.parse_name()
    directives = self.parse_const_directives()
    values = self.parse_enum_values_definition()
    if not (directives or values):
        raise self.unexpected()
    return EnumTypeExtensionNode(name=name, directives=directives, values=values, loc=self.loc(start))