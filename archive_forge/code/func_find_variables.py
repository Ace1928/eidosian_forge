import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def find_variables(fstruct, fs_class='default'):
    """
    :return: The set of variables used by this feature structure.
    :rtype: set(Variable)
    """
    if fs_class == 'default':
        fs_class = _default_fs_class(fstruct)
    return _variables(fstruct, set(), fs_class, set())