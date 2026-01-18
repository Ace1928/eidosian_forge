from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def precision_numerics_many_significant_digits(self):
    """target backend supports values with many digits on both sides,
        such as 319438950232418390.273596, 87673.594069654243

        """
    return exclusions.closed()