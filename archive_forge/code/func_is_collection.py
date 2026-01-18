from typing import (
def is_collection(value: Any) -> bool:
    """Check if value is a collection, but not a string or a mapping."""
    return isinstance(value, collection_types) and (not isinstance(value, not_iterable_types))