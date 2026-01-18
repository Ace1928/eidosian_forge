from __future__ import annotations
import typing
from urllib.parse import parse_qs, unquote
import idna
from ._types import QueryParamTypes, RawURL, URLTypes
from ._urlparse import urlencode, urlparse
from ._utils import primitive_value_to_str
@property
def is_absolute_url(self) -> bool:
    """
        Return `True` for absolute URLs such as 'http://example.com/path',
        and `False` for relative URLs such as '/path'.
        """
    return bool(self._uri_reference.scheme and self._uri_reference.host)