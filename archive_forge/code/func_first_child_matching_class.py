import sys
import os
import re
import warnings
import types
import unicodedata
def first_child_matching_class(self, childclass, start=0, end=sys.maxsize):
    """
        Return the index of the first child whose class exactly matches.

        Parameters:

        - `childclass`: A `Node` subclass to search for, or a tuple of `Node`
          classes. If a tuple, any of the classes may match.
        - `start`: Initial index to check.
        - `end`: Initial index to *not* check.
        """
    if not isinstance(childclass, tuple):
        childclass = (childclass,)
    for index in range(start, min(len(self), end)):
        for c in childclass:
            if isinstance(self[index], c):
                return index
    return None