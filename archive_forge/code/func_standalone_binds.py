from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def standalone_binds(self):
    """target database/driver supports bound parameters as column
        expressions without being in the context of a typed column.
        """
    return exclusions.open()