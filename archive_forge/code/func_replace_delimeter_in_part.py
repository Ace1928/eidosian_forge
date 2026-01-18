import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def replace_delimeter_in_part(p: str) -> str:
    """Replace hyphens with underscores (or vice versa) except when at the start."""
    if get_delimeter() == '-':
        num_underscore_prefix = 0
        for i in range(len(p)):
            if p[i] == '_':
                num_underscore_prefix += 1
            else:
                break
        p = '_' * num_underscore_prefix + p[num_underscore_prefix:].replace('_', '-')
    else:
        p = p.replace('-', '_')
    return p