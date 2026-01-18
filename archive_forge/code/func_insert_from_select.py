from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def insert_from_select(self):
    """target platform supports INSERT from a SELECT."""
    return exclusions.open()