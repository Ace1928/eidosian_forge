from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def on_update_or_deferrable_fks(self):
    return exclusions.only_if(lambda: self.on_update_cascade.enabled or self.deferrable_fks.enabled)