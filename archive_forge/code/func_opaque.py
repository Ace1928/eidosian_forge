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
@classmethod
def opaque(cls, contents: D) -> Resource[D]:
    """
        Create an opaque `Resource` -- i.e. one with opaque specification.

        See `Specification.OPAQUE` for details.
        """
    return Specification.OPAQUE.create_resource(contents=contents)