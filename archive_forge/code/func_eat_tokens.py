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
def eat_tokens(self, predicate):
    """Parse input from tokens while a given condition is met."""
    content = ''
    prev_loc = self._loc
    tok = None
    for tok in self.takewhile(predicate, advance=False):
        content += self._space_between(prev_loc, tok.start)
        content += tok.src
        prev_loc = tok.end
    if tok:
        self._loc = tok.end
    return content