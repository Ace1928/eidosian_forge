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
def tenfn(self, tens, units, mindex=0) -> str:
    if tens != 1:
        tens_part = ten[tens]
        if tens and units:
            hyphen = '-'
        else:
            hyphen = ''
        unit_part = unit[units]
        mill_part = self.millfn(mindex)
        return f'{tens_part}{hyphen}{unit_part}{mill_part}'
    return f'{teen[units]}{mill[mindex]}'