from __future__ import annotations
import abc
import zlib
import hashlib
from lazyops.types import BaseModel
from lazyops.utils.logs import logger
from lazyops.utils.pooler import ThreadPooler
from typing import Any, Optional, Union, Dict, TypeVar, TYPE_CHECKING
def decompress_value(self, value: Union[str, bytes], **kwargs) -> Union[str, bytes]:
    """
        Decompresses the value
        """
    if not self.compression_enabled:
        return value
    try:
        value = self.compressor.decompress(value, **kwargs)
    except Exception as e:
        if self.enable_deprecation_support or self.previous_compressor is not None:
            value = self.deprecated_decompress_value(value, **kwargs)
    if value is not None and (not self.binary):
        value = value.decode(self.encoding)
    return value