from __future__ import annotations
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Match
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import compat
from .langhelpers import _hash_limit_string
from .langhelpers import _warnings_warn
from .langhelpers import decorator
from .langhelpers import inject_docstring_text
from .langhelpers import inject_param_text
from .. import exc
def deprecated_cls(version: str, message: str, constructor: Optional[str]='__init__') -> Callable[[Type[_T]], Type[_T]]:
    header = '.. deprecated:: %s %s' % (version, message or '')

    def decorate(cls: Type[_T]) -> Type[_T]:
        return _decorate_cls_with_warning(cls, constructor, exc.SADeprecationWarning, message % dict(func=constructor), version, header)
    return decorate