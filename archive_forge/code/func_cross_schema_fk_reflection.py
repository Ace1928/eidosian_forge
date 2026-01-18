from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def cross_schema_fk_reflection(self):
    """target system must support reflection of inter-schema
        foreign keys"""
    return exclusions.closed()