from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def parse_upgrade_protocol(header: str, pos: int, header_name: str) -> Tuple[UpgradeProtocol, int]:
    """
    Parse an Upgrade protocol from ``header`` at the given position.

    Return the protocol value and the new position.

    Raises:
        InvalidHeaderFormat: on invalid inputs.

    """
    match = _protocol_re.match(header, pos)
    if match is None:
        raise exceptions.InvalidHeaderFormat(header_name, 'expected protocol', header, pos)
    return (cast(UpgradeProtocol, match.group()), match.end())