from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def reflect_tables_no_columns(self):
    """target database supports creation and reflection of tables with no
        columns, or at least tables that seem to have no columns."""
    return exclusions.closed()