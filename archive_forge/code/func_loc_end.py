from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import contextlib
import itertools
import tokenize
from six import StringIO
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def loc_end(self):
    """Get the end column of the current location parsed to."""
    if self._i < 0:
        return (1, 0)
    return self._tokens[self._i].end