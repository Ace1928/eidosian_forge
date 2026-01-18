from __future__ import annotations
from collections.abc import Iterable, Iterator, Sequence
from enum import Enum
from typing import Any, Callable, ClassVar, Generic, Protocol, TypeVar
from urllib.parse import unquote, urldefrag, urljoin
from attrs import evolve, field
from rpds import HashTrieMap, HashTrieSet, List
from referencing import exceptions
from referencing._attrs import frozen
from referencing.typing import URI, Anchor as AnchorType, D, Mapping, Retrieve
def in_subresource(self, subresource: Resource[D]) -> Resolver[D]:
    """
        Create a resolver for a subresource (which may have a new base URI).
        """
    id = subresource.id()
    if id is None:
        return self
    return evolve(self, base_uri=urljoin(self._base_uri, id))