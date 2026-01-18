import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def retract_bindings(self, bindings):
    """:see: ``nltk.featstruct.retract_bindings()``"""
    return retract_bindings(self, bindings)