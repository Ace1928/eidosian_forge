from __future__ import annotations
from collections import defaultdict
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable
from pydantic_core import CoreSchema, PydanticCustomError, to_jsonable_python
from pydantic_core import core_schema as cs
from ._fields import PydanticMetadata
def check_metadata(metadata: dict[str, Any], allowed: Iterable[str], source_type: Any) -> None:
    """A small utility function to validate that the given metadata can be applied to the target.
    More than saving lines of code, this gives us a consistent error message for all of our internal implementations.

    Args:
        metadata: A dict of metadata.
        allowed: An iterable of allowed metadata.
        source_type: The source type.

    Raises:
        TypeError: If there is metadatas that can't be applied on source type.
    """
    unknown = metadata.keys() - set(allowed)
    if unknown:
        raise TypeError(f'The following constraints cannot be applied to {source_type!r}: {', '.join([f'{k!r}' for k in unknown])}')