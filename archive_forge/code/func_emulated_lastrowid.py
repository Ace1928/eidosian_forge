from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def emulated_lastrowid(self):
    """target dialect retrieves cursor.lastrowid, or fetches
        from a database-side function after an insert() construct executes,
        within the get_lastrowid() method.

        Only dialects that "pre-execute", or need RETURNING to get last
        inserted id, would return closed/fail/skip for this.

        """
    return exclusions.closed()