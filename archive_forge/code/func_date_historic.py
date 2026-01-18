from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def date_historic(self):
    """target dialect supports representation of Python
        datetime.datetime() objects with historic (pre 1970) values."""
    return exclusions.closed()