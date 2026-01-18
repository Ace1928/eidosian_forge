import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
def process_sect(name: str, args: T.List[T.Any]):
    if args:
        parts.append('')
        parts.append(name)
        parts.append('-' * len(parts[-1]))
        for arg in args:
            process_one(arg)