from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import ast
import contextlib
import functools
import itertools
import six
from six.moves import zip
import sys
from pasta.base import ast_constants
from pasta.base import ast_utils
from pasta.base import formatting as fmt
from pasta.base import token_generator
def statement(f):
    """Decorates a function where the node is a statement."""
    return _gen_wrapper(f, scope=False, max_suffix_lines=1, semicolon=True, comment=True, statement=True)