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
def install_reactor(reactor_path: str, event_loop_path: Optional[str]=None) -> None:
    """Installs the :mod:`~twisted.internet.reactor` with the specified
    import path. Also installs the asyncio event loop with the specified import
    path if the asyncio reactor is enabled"""
    reactor_class = load_object(reactor_path)
    if reactor_class is asyncioreactor.AsyncioSelectorReactor:
        set_asyncio_event_loop_policy()
        with suppress(error.ReactorAlreadyInstalledError):
            event_loop = set_asyncio_event_loop(event_loop_path)
            asyncioreactor.install(eventloop=event_loop)
    else:
        *module, _ = reactor_path.split('.')
        installer_path = module + ['install']
        installer = load_object('.'.join(installer_path))
        with suppress(error.ReactorAlreadyInstalledError):
            installer()