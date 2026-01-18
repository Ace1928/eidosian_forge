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
def py__name__(self):
    if not _is_class_instance(self._obj) or inspect.ismethoddescriptor(self._obj):
        cls = self._obj
    else:
        try:
            cls = self._obj.__class__
        except AttributeError:
            return None
    try:
        return cls.__name__
    except AttributeError:
        return None