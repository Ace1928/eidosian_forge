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
def hundsub(self, mo: Match) -> str:
    ret = self.hundfn(int(mo.group(1)), int(mo.group(2)), int(mo.group(3)), self.mill_count)
    self.mill_count += 1
    return ret