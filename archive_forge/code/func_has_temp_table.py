from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def has_temp_table(self):
    """target dialect supports checking a single temp table name"""
    return exclusions.closed()