import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def read_sym_value(self, s, position, reentrances, match):
    val, end = (match.group(), match.end())
    return (self._SYM_CONSTS.get(val, val), end)