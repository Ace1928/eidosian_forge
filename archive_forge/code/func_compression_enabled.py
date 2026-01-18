from __future__ import annotations
import abc
import zlib
import hashlib
from lazyops.types import BaseModel
from lazyops.utils.logs import logger
from lazyops.utils.pooler import ThreadPooler
from typing import Any, Optional, Union, Dict, TypeVar, TYPE_CHECKING
@property
def compression_enabled(self) -> bool:
    """
        Returns if compression is enabled
        """
    return self.compressor is not None