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
def set_asyncio_event_loop(event_loop_path: Optional[str]) -> AbstractEventLoop:
    """Sets and returns the event loop with specified import path."""
    if event_loop_path is not None:
        event_loop_class: Type[AbstractEventLoop] = load_object(event_loop_path)
        event_loop = event_loop_class()
        asyncio.set_event_loop(event_loop)
    else:
        try:
            with catch_warnings():
                filterwarnings('ignore', message='There is no current event loop', category=DeprecationWarning)
                event_loop = asyncio.get_event_loop()
        except RuntimeError:
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
    return event_loop