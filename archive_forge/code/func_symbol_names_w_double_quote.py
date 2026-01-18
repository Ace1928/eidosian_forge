from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def symbol_names_w_double_quote(self):
    """Target driver can create tables with a name like 'some " table'"""
    return exclusions.open()