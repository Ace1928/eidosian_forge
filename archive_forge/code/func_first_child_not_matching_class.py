import sys
import os
import re
import warnings
import types
import unicodedata
def first_child_not_matching_class(self, childclass, start=0, end=sys.maxsize):
    """
        Return the index of the first child whose class does *not* match.

        Parameters:

        - `childclass`: A `Node` subclass to skip, or a tuple of `Node`
          classes. If a tuple, none of the classes may match.
        - `start`: Initial index to check.
        - `end`: Initial index to *not* check.
        """
    if not isinstance(childclass, tuple):
        childclass = (childclass,)
    for index in range(start, min(len(self), end)):
        for c in childclass:
            if isinstance(self.children[index], c):
                break
        else:
            return index
    return None