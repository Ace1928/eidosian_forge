from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def sequences_optional(self):
    """Target database supports sequences, but also optionally
        as a means of generating new PK values."""
    return exclusions.only_if([lambda config: config.db.dialect.supports_sequences and config.db.dialect.sequences_optional], 'no sequence support, or sequences not optional')