from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def parse_scalar_type_extension(self) -> ScalarTypeExtensionNode:
    """ScalarTypeExtension"""
    start = self._lexer.token
    self.expect_keyword('extend')
    self.expect_keyword('scalar')
    name = self.parse_name()
    directives = self.parse_const_directives()
    if not directives:
        raise self.unexpected()
    return ScalarTypeExtensionNode(name=name, directives=directives, loc=self.loc(start))