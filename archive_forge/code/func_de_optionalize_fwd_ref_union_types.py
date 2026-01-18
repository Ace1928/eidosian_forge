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
def de_optionalize_fwd_ref_union_types(type_: ForwardRef) -> _AnnotationScanType:
    """return the non-optional type for Optional[], Union[None, ...], x|None,
    etc. without de-stringifying forward refs.

    unfortunately this seems to require lots of hardcoded heuristics

    """
    annotation = type_.__forward_arg__
    mm = re.match('^(.+?)\\[(.+)\\]$', annotation)
    if mm:
        if mm.group(1) == 'Optional':
            return ForwardRef(mm.group(2))
        elif mm.group(1) == 'Union':
            elements = re.split(',\\s*', mm.group(2))
            return make_union_type(*[ForwardRef(elem) for elem in elements if elem != 'None'])
        else:
            return type_
    pipe_tokens = re.split('\\s*\\|\\s*', annotation)
    if 'None' in pipe_tokens:
        return ForwardRef('|'.join((p for p in pipe_tokens if p != 'None')))
    return type_