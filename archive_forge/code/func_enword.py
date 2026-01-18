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
def enword(self, num: str, group: int) -> str:
    if group == 1:
        num = DIGIT_GROUP.sub(self.group1sub, num)
    elif group == 2:
        num = TWO_DIGITS.sub(self.group2sub, num)
        num = DIGIT_GROUP.sub(self.group1bsub, num, 1)
    elif group == 3:
        num = THREE_DIGITS.sub(self.group3sub, num)
        num = TWO_DIGITS.sub(self.group2sub, num, 1)
        num = DIGIT_GROUP.sub(self.group1sub, num, 1)
    elif int(num) == 0:
        num = self._number_args['zero']
    elif int(num) == 1:
        num = self._number_args['one']
    else:
        num = num.lstrip().lstrip('0')
        self.mill_count = 0
        mo = THREE_DIGITS_WORD.search(num)
        while mo:
            num = THREE_DIGITS_WORD.sub(self.hundsub, num, 1)
            mo = THREE_DIGITS_WORD.search(num)
        num = TWO_DIGITS_WORD.sub(self.tensub, num, 1)
        num = ONE_DIGIT_WORD.sub(self.unitsub, num, 1)
    return num