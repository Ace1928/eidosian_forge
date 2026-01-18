from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def parse_upgrade(header: str) -> List[UpgradeProtocol]:
    """
    Parse an ``Upgrade`` header.

    Return a list of HTTP protocols.

    Args:
        header: value of the ``Upgrade`` header.

    Raises:
        InvalidHeaderFormat: on invalid inputs.

    """
    return parse_list(parse_upgrade_protocol, header, 0, 'Upgrade')