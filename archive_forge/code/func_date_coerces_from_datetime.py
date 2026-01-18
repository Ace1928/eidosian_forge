from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def date_coerces_from_datetime(self):
    """target dialect accepts a datetime object as the target
        of a date column."""
    return exclusions.open()