import asyncio
import zlib
from concurrent.futures import Executor
from typing import Optional, cast
def encoding_to_mode(encoding: Optional[str]=None, suppress_deflate_header: bool=False) -> int:
    if encoding == 'gzip':
        return 16 + zlib.MAX_WBITS
    return -zlib.MAX_WBITS if suppress_deflate_header else zlib.MAX_WBITS