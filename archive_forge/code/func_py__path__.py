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
def py__path__(self):
    paths = getattr(self._obj, '__path__', None)
    if not isinstance(paths, list) or not all((isinstance(p, str) for p in paths)):
        return None
    return paths