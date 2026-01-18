from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def fetch_offset_with_options(self):
    """backend supports the offset when using fetch first with percent
        or ties. basically this is "not mssql"
        """
    return exclusions.closed()