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
def py__simple_getitem__(self, index, *, safe=True):
    if safe and type(self._obj) not in ALLOWED_GETITEM_TYPES:
        return None
    return self._create_access_path(self._obj[index])