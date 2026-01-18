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
def py__file__(self) -> Optional[Path]:
    try:
        return Path(self._obj.__file__)
    except AttributeError:
        return None