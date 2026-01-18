from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def parse_extension_item_param(header: str, pos: int, header_name: str) -> Tuple[ExtensionParameter, int]:
    """
    Parse a single extension parameter from ``header`` at the given position.

    Return a ``(name, value)`` pair and the new position.

    Raises:
        InvalidHeaderFormat: on invalid inputs.

    """
    name, pos = parse_token(header, pos, header_name)
    pos = parse_OWS(header, pos)
    value: Optional[str] = None
    if peek_ahead(header, pos) == '=':
        pos = parse_OWS(header, pos + 1)
        if peek_ahead(header, pos) == '"':
            pos_before = pos
            value, pos = parse_quoted_string(header, pos, header_name)
            if _token_re.fullmatch(value) is None:
                raise exceptions.InvalidHeaderFormat(header_name, 'invalid quoted header content', header, pos_before)
        else:
            value, pos = parse_token(header, pos, header_name)
        pos = parse_OWS(header, pos)
    return ((name, value), pos)