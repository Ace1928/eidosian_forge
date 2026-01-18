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
def defnoun(self, singular: Optional[Word], plural: Optional[Word]) -> int:
    """
        Set the noun plural of singular to plural.

        """
    self.checkpat(singular)
    self.checkpatplural(plural)
    self.pl_sb_user_defined.extend((singular, plural))
    self.si_sb_user_defined.extend((plural, singular))
    return 1