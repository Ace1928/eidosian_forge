from __future__ import annotations
import numpy as np
def is_valid_chunk_type(type):
    """Check if given type is a valid chunk and downcast array type"""
    try:
        return type in _HANDLED_CHUNK_TYPES or issubclass(type, tuple(_HANDLED_CHUNK_TYPES))
    except TypeError:
        return False