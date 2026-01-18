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
def checkpat(self, pattern: Optional[Word]) -> None:
    """
        check for errors in a regex pattern
        """
    if pattern is None:
        return
    try:
        re.match(pattern, '')
    except re.error as err:
        raise BadUserDefinedPatternError(pattern) from err