import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def read_var_value(self, s, position, reentrances, match):
    return (Variable(match.group()), match.end())