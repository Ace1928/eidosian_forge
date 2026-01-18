from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def no_lastrowid_support(self):
    """the opposite of supports_lastrowid"""
    return exclusions.only_if([lambda config: not config.db.dialect.postfetch_lastrowid])