from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
def merge_seqs(seqs: list[deque[type[Any]]]) -> Iterable[type[Any]]:
    while True:
        non_empty = [seq for seq in seqs if seq]
        if not non_empty:
            return
        candidate: type[Any] | None = None
        for seq in non_empty:
            candidate = seq[0]
            not_head = [s for s in non_empty if candidate in islice(s, 1, None)]
            if not_head:
                candidate = None
            else:
                break
        if not candidate:
            raise TypeError('Inconsistent hierarchy, no C3 MRO is possible')
        yield candidate
        for seq in non_empty:
            if seq[0] == candidate:
                seq.popleft()