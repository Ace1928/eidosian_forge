from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def dialect_level_isolation_level_param(self):
    """test that the dialect allows the 'isolation_level' argument
        to be handled by DefaultDialect"""

    def go(config):
        try:
            e = create_engine(config.db.url, isolation_level='READ COMMITTED')
        except:
            return False
        else:
            return e.dialect._on_connect_isolation_level == 'READ COMMITTED'
    return exclusions.only_if(go)