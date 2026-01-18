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
def group1sub(self, mo: Match) -> str:
    units = int(mo.group(1))
    if units == 1:
        return f' {self._number_args['one']}, '
    elif units:
        return f'{unit[units]}, '
    else:
        return f' {self._number_args['zero']}, '