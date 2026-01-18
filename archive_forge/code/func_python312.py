from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def python312(self):
    return exclusions.only_if(lambda: util.py312, 'Python 3.12 or above required')