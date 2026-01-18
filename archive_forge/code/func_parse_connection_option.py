from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def parse_connection_option(header: str, pos: int, header_name: str) -> Tuple[ConnectionOption, int]:
    """
    Parse a Connection option from ``header`` at the given position.

    Return the protocol value and the new position.

    Raises:
        InvalidHeaderFormat: on invalid inputs.

    """
    item, pos = parse_token(header, pos, header_name)
    return (cast(ConnectionOption, item), pos)