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
def py__iter__list(self):
    try:
        iter_method = self._obj.__iter__
    except AttributeError:
        return None
    else:
        p = DirectObjectAccess(self._inference_state, iter_method).get_return_annotation()
        if p is not None:
            return [p]
    if type(self._obj) not in ALLOWED_GETITEM_TYPES:
        return []
    lst = []
    for i, part in enumerate(self._obj):
        if i > 20:
            break
        lst.append(self._create_access_path(part))
    return lst