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
def from_contents(cls, contents: D, default_specification: type[Specification[D]] | Specification[D]=Specification) -> Resource[D]:
    """
        Create a resource guessing which specification applies to the contents.

        Raises:

            `CannotDetermineSpecification`

                if the given contents don't have any discernible
                information which could be used to guess which
                specification they identify as

        """
    specification = default_specification.detect(contents)
    return specification.create_resource(contents=contents)