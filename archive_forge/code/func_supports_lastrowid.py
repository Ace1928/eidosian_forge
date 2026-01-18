from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def supports_lastrowid(self):
    """target database / driver supports cursor.lastrowid as a means
        of retrieving the last inserted primary key value.

        note that if the target DB supports sequences also, this is still
        assumed to work.  This is a new use case brought on by MariaDB 10.3.

        """
    return exclusions.only_if([lambda config: config.db.dialect.postfetch_lastrowid])