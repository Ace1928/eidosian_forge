from __future__ import annotations
import abc
import zlib
import hashlib
from lazyops.types import BaseModel
from lazyops.utils.logs import logger
from lazyops.utils.pooler import ThreadPooler
from typing import Any, Optional, Union, Dict, TypeVar, TYPE_CHECKING
def fetch_object_classname(self, obj: ObjectValue) -> str:
    """
        Fetches the object classname
        """
    return f'{obj.__class__.__module__}.{obj.__class__.__name__}'