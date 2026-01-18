from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def precision_numerics_general(self):
    """target backend has general support for moderately high-precision
        numerics."""
    return exclusions.open()