from __future__ import annotations
import typing
from urllib.parse import parse_qs, unquote
import idna
from ._types import QueryParamTypes, RawURL, URLTypes
from ._urlparse import urlencode, urlparse
from ._utils import primitive_value_to_str
@property
def raw_host(self) -> bytes:
    """
        The raw bytes representation of the URL host.
        Always normalized to lowercase, and IDNA encoded.

        Examples:

        url = httpx.URL("http://www.EXAMPLE.org")
        assert url.raw_host == b"www.example.org"

        url = httpx.URL("http://中国.icom.museum")
        assert url.raw_host == b"xn--fiqs8s.icom.museum"

        url = httpx.URL("http://xn--fiqs8s.icom.museum")
        assert url.raw_host == b"xn--fiqs8s.icom.museum"

        url = httpx.URL("https://[::ffff:192.168.0.1]")
        assert url.raw_host == b"::ffff:192.168.0.1"
        """
    return self._uri_reference.host.encode('ascii')