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
def group2sub(self, mo: Match) -> str:
    tens = int(mo.group(1))
    units = int(mo.group(2))
    if tens:
        return f'{self.tenfn(tens, units)}, '
    if units:
        return f' {self._number_args['zero']} {unit[units]}, '
    return f' {self._number_args['zero']} {self._number_args['zero']}, '