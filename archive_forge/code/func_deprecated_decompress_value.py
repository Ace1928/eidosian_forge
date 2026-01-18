from __future__ import annotations
import abc
import zlib
import hashlib
from lazyops.types import BaseModel
from lazyops.utils.logs import logger
from lazyops.utils.pooler import ThreadPooler
from typing import Any, Optional, Union, Dict, TypeVar, TYPE_CHECKING
def deprecated_decompress_value(self, value: Union[str, bytes], **kwargs) -> Optional[Union[str, bytes]]:
    """
        Attempts to decompress the value using the deprecated compressor
        """
    e = None
    attempt_msg = f'{self.name}'
    if self.previous_compressor is not None:
        try:
            return self.previous_compressor.decompress(value)
        except Exception as e:
            attempt_msg += f'-> {self.previous_compressor.name}'
    try:
        return zlib.decompress(value)
    except Exception as e:
        attempt_msg += ' -> ZLib'
        logger.trace(f'[{attempt_msg}] Error in Decompression: {str(value)[:100]}', e)
        if self.raise_errors:
            raise e
        return None