from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def empty_inserts_executemany(self):
    """target platform supports INSERT with no values, i.e.
        INSERT DEFAULT VALUES or equivalent, within executemany()"""
    return self.empty_inserts