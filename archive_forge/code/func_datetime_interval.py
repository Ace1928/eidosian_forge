from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def datetime_interval(self):
    """target dialect supports rendering of a datetime.timedelta as a
        literal string, e.g. via the TypeEngine.literal_processor() method.

        """
    return exclusions.closed()