import functools
import math
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from ipaddress import ip_address
from urllib.parse import SplitResult, parse_qsl, quote, urljoin, urlsplit, urlunsplit
import idna
from multidict import MultiDict, MultiDictProxy
from ._quoting import _Quoter, _Unquoter
@property
def explicit_port(self):
    """Port part of URL, without scheme-based fallback.

        None for relative URLs or URLs without explicit port.

        """
    return self._val.port