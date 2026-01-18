import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def make_subparser_dest(name: str) -> str:
    return f'{name} (positional)'