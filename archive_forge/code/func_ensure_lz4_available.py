from .base import BaseCompress
from typing import Union
def ensure_lz4_available():
    """
    Ensure LZ4 is available
    """
    global _lz4_available, lz4
    if _lz4_available is False:
        from lazyops.utils.imports import resolve_missing
        resolve_missing('lz4', required=True)
        import lz4.frame
        _lz4_available = True
        globals()['lz4'] = lz4