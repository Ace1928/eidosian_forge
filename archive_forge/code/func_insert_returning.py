from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def insert_returning(self):
    """target platform supports INSERT ... RETURNING."""
    return exclusions.only_if(lambda config: config.db.dialect.insert_returning, "%(database)s %(does_support)s 'INSERT ... RETURNING'")