from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def schema_create_delete(self):
    """target database supports schema create and dropped with
        'CREATE SCHEMA' and 'DROP SCHEMA'"""
    return exclusions.closed()