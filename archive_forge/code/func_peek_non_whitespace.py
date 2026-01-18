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
def peek_non_whitespace(self):
    """Get the next non-whitespace token without advancing."""
    return self.peek_conditional(lambda t: t.type not in FORMATTING_TOKENS)