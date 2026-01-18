from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def comment_reflection(self):
    """Indicates if the database support table comment reflection"""
    return exclusions.closed()