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
def get_si_pron(thecase, word, gender) -> str:
    try:
        sing = si_pron[thecase][word]
    except KeyError:
        raise
    try:
        return sing[gender]
    except TypeError:
        return cast(str, sing)