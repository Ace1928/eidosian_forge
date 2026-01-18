from __future__ import annotations
import abc
import zlib
import hashlib
from lazyops.types import BaseModel
from lazyops.utils.logs import logger
from lazyops.utils.pooler import ThreadPooler
from typing import Any, Optional, Union, Dict, TypeVar, TYPE_CHECKING
def create_obj_hash(obj: ObjectValue) -> str:
    """
    Creates a hash for the object
    """
    if isinstance(obj, BaseModel) or hasattr(obj, 'model_dump'):
        if _xxhash_available:
            return xxhash.xxh64(obj.model_dump_json()).hexdigest()
        return hashlib.sha256(obj.model_dump_json().encode()).hexdigest()
    if isinstance(obj, dict):
        if _xxhash_available:
            return xxhash.xxh64(str(obj).encode()).hexdigest()
        return hashlib.sha256(str(obj).encode()).hexdigest()
    if _xxhash_available:
        return xxhash.xxh64(str(obj).encode()).hexdigest()
    return hashlib.sha256(str(obj).encode()).hexdigest()