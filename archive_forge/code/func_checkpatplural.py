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
def checkpatplural(self, pattern: Optional[Word]) -> None:
    """
        check for errors in a regex replace pattern
        """
    return