from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_enum_values_definition(self) -> List[EnumValueDefinitionNode]:
    """EnumValuesDefinition: {EnumValueDefinition+}"""
    return self.optional_many(TokenKind.BRACE_L, self.parse_enum_value_definition, TokenKind.BRACE_R)