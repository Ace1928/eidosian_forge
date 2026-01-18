from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def boolean_col_expressions(self):
    """Target database must support boolean expressions as columns"""
    return exclusions.closed()