import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def multi_metavar_from_single(single: str) -> str:
    if len(strip_ansi_sequences(single)) >= 32:
        return f'[{single} [...]]'
    else:
        return f'[{single} [{single} ...]]'