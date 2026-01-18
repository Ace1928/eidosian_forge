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
@typechecked
def defan(self, pattern: Optional[Word]) -> int:
    """
        Define the indefinite article as 'an' for words matching pattern.

        """
    self.checkpat(pattern)
    self.A_a_user_defined.extend((pattern, 'an'))
    return 1