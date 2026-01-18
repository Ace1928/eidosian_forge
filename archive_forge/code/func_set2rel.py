import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
def set2rel(s):
    """
    Convert a set containing individuals (strings or numbers) into a set of
    unary tuples. Any tuples of strings already in the set are passed through
    unchanged.

    For example:
      - set(['a', 'b']) => set([('a',), ('b',)])
      - set([3, 27]) => set([('3',), ('27',)])

    :type s: set
    :rtype: set of tuple of str
    """
    new = set()
    for elem in s:
        if isinstance(elem, str):
            new.add((elem,))
        elif isinstance(elem, int):
            new.add(str(elem))
        else:
            new.add(elem)
    return new