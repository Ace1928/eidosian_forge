from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def schemas(self):
    """Target database must support external schemas, and have one
        named 'test_schema'."""
    return only_on(lambda config: config.db.dialect.supports_schemas)