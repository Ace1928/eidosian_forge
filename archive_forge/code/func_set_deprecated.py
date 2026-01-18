from typing import Optional, TypeVar
def set_deprecated(obj: T) -> T:
    """Explicitly tag an object as deprecated for the doc generator."""
    setattr(obj, _DEPRECATED, None)
    return obj