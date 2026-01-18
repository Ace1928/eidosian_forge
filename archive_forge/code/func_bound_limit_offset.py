from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def bound_limit_offset(self):
    """target database can render LIMIT and/or OFFSET using a bound
        parameter
        """
    return exclusions.open()