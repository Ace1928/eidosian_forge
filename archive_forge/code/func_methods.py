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
def methods(self):
    if self._cls is None:
        return []
    return [name for name, func in inspect.getmembers(self._cls) if (not name.startswith('_') or name in self.extra_public_methods) and isinstance(func, Callable) and self._is_show_member(name)]