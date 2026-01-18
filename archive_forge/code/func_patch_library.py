from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def patch_library(self):

    def check_lib():
        try:
            __import__('patch')
        except ImportError:
            return False
        else:
            return True
    return exclusions.only_if(check_lib, 'patch library needed')