import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def rename_variables(self, vars=None, used_vars=(), new_vars=None):
    """:see: ``nltk.featstruct.rename_variables()``"""
    return rename_variables(self, vars, used_vars, new_vars)