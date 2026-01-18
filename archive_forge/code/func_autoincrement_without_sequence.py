from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def autoincrement_without_sequence(self):
    """If autoincrement=True on a column does not require an explicit
        sequence. This should be false only for oracle.
        """
    return exclusions.open()