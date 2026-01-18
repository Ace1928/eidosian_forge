import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def parse_item_name(text):
    """Match ':role:`name`' or 'name'."""
    m = self._func_rgx.match(text)
    if not m:
        raise ParseError('%s is not a item name' % text)
    role = m.group('role')
    name = m.group('name') if role else m.group('name2')
    return (name, role, m.end())