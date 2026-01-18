from __future__ import annotations
from collections.abc import Sequence, Set
from typing import Any, Iterable, Union
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing._attrs import frozen
from referencing._core import (
from referencing.typing import URI, Anchor as AnchorType, Mapping
def lookup_recursive_ref(resolver: _Resolver[Schema]) -> _Resolved[Schema]:
    """
    Recursive references (via recursive anchors), present only in draft 2019.

    As per the 2019 specification (§ 8.2.4.2.1), only the ``#`` recursive
    reference is supported (and is therefore assumed to be the relevant
    reference).
    """
    resolved = resolver.lookup('#')
    if isinstance(resolved.contents, Mapping) and resolved.contents.get('$recursiveAnchor'):
        for uri, _ in resolver.dynamic_scope():
            next_resolved = resolver.lookup(uri)
            if not isinstance(next_resolved.contents, Mapping) or not next_resolved.contents.get('$recursiveAnchor'):
                break
            resolved = next_resolved
    return resolved