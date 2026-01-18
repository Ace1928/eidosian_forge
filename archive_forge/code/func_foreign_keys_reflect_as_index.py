from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def foreign_keys_reflect_as_index(self):
    """Target database creates an index that's reflected for
        foreign keys."""
    return exclusions.closed()