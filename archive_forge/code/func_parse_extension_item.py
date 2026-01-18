from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def parse_extension_item(header: str, pos: int, header_name: str) -> Tuple[ExtensionHeader, int]:
    """
    Parse an extension definition from ``header`` at the given position.

    Return an ``(extension name, parameters)`` pair, where ``parameters`` is a
    list of ``(name, value)`` pairs, and the new position.

    Raises:
        InvalidHeaderFormat: on invalid inputs.

    """
    name, pos = parse_token(header, pos, header_name)
    pos = parse_OWS(header, pos)
    parameters = []
    while peek_ahead(header, pos) == ';':
        pos = parse_OWS(header, pos + 1)
        parameter, pos = parse_extension_item_param(header, pos, header_name)
        parameters.append(parameter)
    return ((cast(ExtensionName, name), parameters), pos)