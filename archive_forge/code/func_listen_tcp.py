import asyncio
import sys
from asyncio import AbstractEventLoop, AbstractEventLoopPolicy
from contextlib import suppress
from typing import Any, Callable, Dict, Optional, Sequence, Type
from warnings import catch_warnings, filterwarnings, warn
from twisted.internet import asyncioreactor, error
from twisted.internet.base import DelayedCall
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.misc import load_object
def listen_tcp(portrange, host, factory):
    """Like reactor.listenTCP but tries different ports in a range."""
    from twisted.internet import reactor
    if len(portrange) > 2:
        raise ValueError(f'invalid portrange: {portrange}')
    if not portrange:
        return reactor.listenTCP(0, factory, interface=host)
    if not hasattr(portrange, '__iter__'):
        return reactor.listenTCP(portrange, factory, interface=host)
    if len(portrange) == 1:
        return reactor.listenTCP(portrange[0], factory, interface=host)
    for x in range(portrange[0], portrange[1] + 1):
        try:
            return reactor.listenTCP(x, factory, interface=host)
        except error.CannotListenError:
            if x == portrange[1]:
                raise