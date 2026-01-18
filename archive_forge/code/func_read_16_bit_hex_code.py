from typing import List, NamedTuple, Optional
from ..error import GraphQLSyntaxError
from .ast import Token
from .block_string import dedent_block_string_lines
from .character_classes import is_digit, is_name_start, is_name_continue
from .source import Source
from .token_kind import TokenKind
def read_16_bit_hex_code(body: str, position: int) -> int:
    """Read a 16bit hexadecimal string and return its positive integer value (0-65535).

    Reads four hexadecimal characters and returns the positive integer that 16bit
    hexadecimal string represents. For example, "000f" will return 15, and "dead"
    will return 57005.

    Returns a negative number if any char was not a valid hexadecimal digit.
    """
    return read_hex_digit(body[position]) << 12 | read_hex_digit(body[position + 1]) << 8 | read_hex_digit(body[position + 2]) << 4 | read_hex_digit(body[position + 3])