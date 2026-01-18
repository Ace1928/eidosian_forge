from __future__ import annotations
from configparser import ConfigParser
import io
import os
import sys
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
def formatannotation_fwdref(annotation: Any, base_module: Optional[Any]=None) -> str:
    """vendored from python 3.7"""
    if isinstance(annotation, str):
        return annotation
    if getattr(annotation, '__module__', None) == 'typing':
        return repr(annotation).replace('typing.', '').replace('~', '')
    if isinstance(annotation, type):
        if annotation.__module__ in ('builtins', base_module):
            return repr(annotation.__qualname__)
        return annotation.__module__ + '.' + annotation.__qualname__
    elif isinstance(annotation, typing.TypeVar):
        return repr(annotation).replace('~', '')
    return repr(annotation).replace('~', '')