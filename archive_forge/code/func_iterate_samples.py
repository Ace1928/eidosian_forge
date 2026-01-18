from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, TypeVar
def iterate_samples(self, collection: Iterable[T]) -> Iterable[T]:
    if self is CollectionCheckStrategy.FIRST_ITEM:
        try:
            return [next(iter(collection))]
        except StopIteration:
            return ()
    else:
        return collection