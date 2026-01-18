import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
def partition_word(self, text: str) -> Tuple[str, str, str]:
    mo = PARTITION_WORD.search(text)
    if mo:
        return (mo.group(1), mo.group(2), mo.group(3))
    else:
        return ('', '', '')