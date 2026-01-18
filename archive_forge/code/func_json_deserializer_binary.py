from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def json_deserializer_binary(self):
    """indicates if the json_deserializer function is called with bytes"""
    return exclusions.closed()