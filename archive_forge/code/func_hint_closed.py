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
def hint_closed(self):
    """Indicates closing a group of parentheses or brackets."""
    self._hints -= 1
    if self._hints < 0:
        raise ValueError('Hint value negative')