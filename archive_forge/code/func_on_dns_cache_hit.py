from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, Optional, Protocol, Type, TypeVar
import attr
from aiosignal import Signal
from multidict import CIMultiDict
from yarl import URL
from .client_reqrep import ClientResponse
@property
def on_dns_cache_hit(self) -> 'Signal[_SignalCallback[TraceDnsCacheHitParams]]':
    return self._on_dns_cache_hit