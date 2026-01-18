from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def python310(self):
    return exclusions.only_if(lambda: util.py310, 'Python 3.10 or above required')