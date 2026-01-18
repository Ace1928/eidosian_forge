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
@rewrite_module
def cache_configure(*, idna_encode_size=_MAXCACHE, idna_decode_size=_MAXCACHE):
    global _idna_decode, _idna_encode
    _idna_encode = functools.lru_cache(idna_encode_size)(_idna_encode.__wrapped__)
    _idna_decode = functools.lru_cache(idna_decode_size)(_idna_decode.__wrapped__)