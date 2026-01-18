from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def expressions_against_unbounded_text(self):
    """target database supports use of an unbounded textual field in a
        WHERE clause."""
    return exclusions.open()