from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def empty_strings_text(self):
    """target database can persist/return an empty string with an
        unbounded text."""
    return exclusions.open()