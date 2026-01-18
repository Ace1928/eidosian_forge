import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def subparser_name_from_type(prefix: str, cls: Type) -> str:
    suffix, use_prefix = _subparser_name_from_type(cls) if cls is not type(None) else ('None', True)
    if len(prefix) == 0 or not use_prefix:
        return suffix
    if get_delimeter() == '-':
        return f'{prefix}:{make_field_name(suffix.split('.'))}'
    else:
        assert get_delimeter() == '_'
        return f'{prefix}:{suffix}'