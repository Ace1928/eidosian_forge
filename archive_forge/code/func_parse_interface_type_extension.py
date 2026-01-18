from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_interface_type_extension(self) -> InterfaceTypeExtensionNode:
    """InterfaceTypeExtension"""
    start = self._lexer.token
    self.expect_keyword('extend')
    self.expect_keyword('interface')
    name = self.parse_name()
    interfaces = self.parse_implements_interfaces()
    directives = self.parse_const_directives()
    fields = self.parse_fields_definition()
    if not (interfaces or directives or fields):
        raise self.unexpected()
    return InterfaceTypeExtensionNode(name=name, interfaces=interfaces, directives=directives, fields=fields, loc=self.loc(start))