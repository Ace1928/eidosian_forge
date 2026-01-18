from __future__ import annotations
import base64
import binascii
import typing as t
from ..http import dump_header
from ..http import parse_dict_header
from ..http import quote_header_value
from .structures import CallbackDict
@classmethod
def from_header(cls, value: str | None) -> te.Self | None:
    """Parse a ``WWW-Authenticate`` header value and return an instance, or ``None``
        if the value is empty.

        :param value: The header value to parse.

        .. versionadded:: 2.3
        """
    if not value:
        return None
    scheme, _, rest = value.partition(' ')
    scheme = scheme.lower()
    rest = rest.strip()
    if '=' in rest.rstrip('='):
        return cls(scheme, parse_dict_header(rest), None)
    return cls(scheme, None, rest)