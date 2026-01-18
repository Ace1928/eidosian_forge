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
def get_asyncio_event_loop_policy() -> AbstractEventLoopPolicy:
    warn('Call to deprecated function scrapy.utils.reactor.get_asyncio_event_loop_policy().\n\nPlease use get_event_loop, new_event_loop and set_event_loop from asyncio instead, as the corresponding policy methods may lead to unexpected behaviour.\nThis function is replaced by set_asyncio_event_loop_policy and is meant to be used only when the reactor is being installed.', category=ScrapyDeprecationWarning, stacklevel=2)
    return _get_asyncio_event_loop_policy()