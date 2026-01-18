from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def multivalues_inserts(self):
    """target database must support multiple VALUES clauses in an
        INSERT statement."""
    return exclusions.skip_if(lambda config: not config.db.dialect.supports_multivalues_insert, 'Backend does not support multirow inserts.')