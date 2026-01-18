from __future__ import annotations
from .base import BaseCompression, logger
from typing import Optional
def check_deps(self):
    """
        Checks for dependencies
        """
    if _zstd_available is False:
        logger.error('zstd is not available. Please install `zstd` or `pyzstd` to use zstd compression')
        raise ImportError('zstd is not available. Please install `zstd` or `pyzstd` to use zstd compression')