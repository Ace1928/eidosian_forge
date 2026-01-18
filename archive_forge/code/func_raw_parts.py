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
@cached_property
def raw_parts(self):
    """A tuple containing encoded *path* parts.

        ('/',) for absolute URLs if *path* is missing.

        """
    path = self._val.path
    if self.is_absolute():
        if not path:
            parts = ['/']
        else:
            parts = ['/'] + path[1:].split('/')
    elif path.startswith('/'):
        parts = ['/'] + path[1:].split('/')
    else:
        parts = path.split('/')
    return tuple(parts)