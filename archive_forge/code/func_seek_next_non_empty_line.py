import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def seek_next_non_empty_line(self):
    for l in self[self._l:]:
        if l.strip():
            break
        else:
            self._l += 1