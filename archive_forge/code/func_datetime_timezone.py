from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def datetime_timezone(self):
    """target dialect supports representation of Python
        datetime.datetime() with tzinfo with DateTime(timezone=True)."""
    return exclusions.closed()