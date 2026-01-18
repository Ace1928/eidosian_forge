from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def parse_subprotocol_item(header: str, pos: int, header_name: str) -> Tuple[Subprotocol, int]:
    """
    Parse a subprotocol from ``header`` at the given position.

    Return the subprotocol value and the new position.

    Raises:
        InvalidHeaderFormat: on invalid inputs.

    """
    item, pos = parse_token(header, pos, header_name)
    return (cast(Subprotocol, item), pos)