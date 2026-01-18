from typing import Callable, Dict, List, Optional, Union, TypeVar, cast
from functools import partial
from .ast import (
from .directive_locations import DirectiveLocation
from .ast import Token
from .lexer import Lexer, is_punctuator_token_kind
from .source import Source, is_source
from .token_kind import TokenKind
from ..error import GraphQLError, GraphQLSyntaxError
def advance_lexer(self) -> None:
    max_tokens = self._max_tokens
    token = self._lexer.advance()
    if max_tokens is not None and token.kind != TokenKind.EOF:
        self._token_counter += 1
        if self._token_counter > max_tokens:
            raise GraphQLSyntaxError(self._lexer.source, token.start, f'Document contains more than {max_tokens} tokens. Parsing aborted.')