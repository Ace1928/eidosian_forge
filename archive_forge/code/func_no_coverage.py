from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def no_coverage(self):
    """Test should be skipped if coverage is enabled.

        This is to block tests that exercise libraries that seem to be
        sensitive to coverage, such as PostgreSQL notice logging.

        """
    return exclusions.skip_if(lambda config: config.options.has_coverage, 'Issues observed when coverage is enabled')