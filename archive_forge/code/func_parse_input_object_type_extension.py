from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_input_object_type_extension(self) -> InputObjectTypeExtensionNode:
    """InputObjectTypeExtension"""
    start = self._lexer.token
    self.expect_keyword('extend')
    self.expect_keyword('input')
    name = self.parse_name()
    directives = self.parse_const_directives()
    fields = self.parse_input_fields_definition()
    if not (directives or fields):
        raise self.unexpected()
    return InputObjectTypeExtensionNode(name=name, directives=directives, fields=fields, loc=self.loc(start))