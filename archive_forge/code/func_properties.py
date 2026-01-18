import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
@property
def properties(self):
    if self._cls is None:
        return []
    return [name for name, func in inspect.getmembers(self._cls) if not name.startswith('_') and (func is None or isinstance(func, property) or inspect.isdatadescriptor(func)) and self._is_show_member(name)]