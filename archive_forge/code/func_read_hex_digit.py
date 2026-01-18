from typing import List, NamedTuple, Optional
from ..error import GraphQLSyntaxError
from .ast import Token
from .block_string import dedent_block_string_lines
from .character_classes import is_digit, is_name_start, is_name_continue
from .source import Source
from .token_kind import TokenKind
def read_hex_digit(char: str) -> int:
    """Read a hexadecimal character and returns its positive integer value (0-15).

    '0' becomes 0, '9' becomes 9
    'A' becomes 10, 'F' becomes 15
    'a' becomes 10, 'f' becomes 15

    Returns -1 if the provided character code was not a valid hexadecimal digit.
    """
    if '0' <= char <= '9':
        return ord(char) - 48
    elif 'A' <= char <= 'F':
        return ord(char) - 55
    elif 'a' <= char <= 'f':
        return ord(char) - 87
    return -1