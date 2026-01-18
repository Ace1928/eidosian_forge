import importlib
import importlib.metadata as importlib_metadata
from functools import lru_cache
import packaging.version
def is_bnb_4bit_available() -> bool:
    if not is_bnb_available():
        return False
    import bitsandbytes as bnb
    return hasattr(bnb.nn, 'Linear4bit')