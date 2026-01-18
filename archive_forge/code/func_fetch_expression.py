from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def fetch_expression(self):
    """backend supports fetch / offset with expression in them, like

        SELECT * FROM some_table
        OFFSET 1 + 1 ROWS FETCH FIRST 1 + 1 ROWS ONLY
        """
    return exclusions.closed()