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
def raw_user(self):
    """Encoded user part of URL.

        None if user is missing.

        """
    ret = self._val.username
    if not ret:
        return None
    return ret