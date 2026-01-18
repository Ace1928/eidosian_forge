from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def expect_optional_keyword(self, value: str) -> bool:
    """Expect the next token optionally to be a given keyword.

        If the next token is a given keyword, return True after advancing the lexer.
        Otherwise, do not change the parser state and return False.
        """
    token = self._lexer.token
    if token.kind == TokenKind.NAME and token.value == value:
        self.advance_lexer()
        return True
    return False