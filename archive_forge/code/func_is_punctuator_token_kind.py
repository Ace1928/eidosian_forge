from typing import List, NamedTuple, Optional
from ..error import GraphQLSyntaxError
from .ast import Token
from .block_string import dedent_block_string_lines
from .character_classes import is_digit, is_name_start, is_name_continue
from .source import Source
from .token_kind import TokenKind
def is_punctuator_token_kind(kind: TokenKind) -> bool:
    """Check whether the given token kind corresponds to a punctuator.

    For internal use only.
    """
    return kind in _punctuator_token_kinds