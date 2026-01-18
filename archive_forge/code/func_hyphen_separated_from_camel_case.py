import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def hyphen_separated_from_camel_case(name: str) -> str:
    out = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))').sub(get_delimeter() + '\\1', name).lower()
    return out