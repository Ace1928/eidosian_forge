from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def implicit_default_schema(self):
    """target system has a strong concept of 'default' schema that can
        be referred to implicitly.

        basically, PostgreSQL.

        """
    return exclusions.closed()