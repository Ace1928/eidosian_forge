from __future__ import annotations
import builtins
import collections.abc as collections_abc
import re
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import ForwardRef
from typing import Generic
from typing import Iterable
from typing import Mapping
from typing import NewType
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import compat
def de_stringify_annotation(cls: Type[Any], annotation: _AnnotationScanType, originating_module: str, locals_: Mapping[str, Any], *, str_cleanup_fn: Optional[Callable[[str, str], str]]=None, include_generic: bool=False, _already_seen: Optional[Set[Any]]=None) -> Type[Any]:
    """Resolve annotations that may be string based into real objects.

    This is particularly important if a module defines "from __future__ import
    annotations", as everything inside of __annotations__ is a string. We want
    to at least have generic containers like ``Mapped``, ``Union``, ``List``,
    etc.

    """
    original_annotation = annotation
    if is_fwd_ref(annotation):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        if str_cleanup_fn:
            annotation = str_cleanup_fn(annotation, originating_module)
        annotation = eval_expression(annotation, originating_module, locals_=locals_, in_class=cls)
    if include_generic and is_generic(annotation) and (not is_literal(annotation)):
        if _already_seen is None:
            _already_seen = set()
        if annotation in _already_seen:
            return original_annotation
        else:
            _already_seen.add(annotation)
        elements = tuple((de_stringify_annotation(cls, elem, originating_module, locals_, str_cleanup_fn=str_cleanup_fn, include_generic=include_generic, _already_seen=_already_seen) for elem in annotation.__args__))
        return _copy_generic_annotation_with(annotation, elements)
    return annotation