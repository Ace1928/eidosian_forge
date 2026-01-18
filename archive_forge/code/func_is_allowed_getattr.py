import inspect
import types
import traceback
import sys
import operator as op
from collections import namedtuple
import warnings
import re
import builtins
import typing
from pathlib import Path
from typing import Optional, Tuple
from jedi.inference.compiled.getattr_static import getattr_static
def is_allowed_getattr(self, name, safe=True) -> Tuple[bool, bool, Optional[AccessPath]]:
    try:
        attr, is_get_descriptor = getattr_static(self._obj, name)
    except AttributeError:
        if not safe:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                try:
                    return (hasattr(self._obj, name), False, None)
                except Exception:
                    pass
        return (False, False, None)
    else:
        if is_get_descriptor and type(attr) not in ALLOWED_DESCRIPTOR_ACCESS:
            if isinstance(attr, property):
                if hasattr(attr.fget, '__annotations__'):
                    a = DirectObjectAccess(self._inference_state, attr.fget)
                    return (True, True, a.get_return_annotation())
            return (True, True, None)
    return (True, False, None)