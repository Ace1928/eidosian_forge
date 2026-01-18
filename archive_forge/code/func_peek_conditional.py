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
def peek_conditional(self, condition):
    """Get the next token of the given type without advancing."""
    return next((t for t in self._tokens[self._i + 1:] if condition(t)), None)