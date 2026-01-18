from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def sane_rowcount_w_returning(self):
    return exclusions.fails_if(lambda config: not config.db.dialect.supports_sane_rowcount_returning, "driver doesn't support 'sane' rowcount when returning is on")