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
def arg_location(tup):
    arg = tup[1]
    if isinstance(arg, ast.keyword):
        arg = arg.value
    return (getattr(arg, 'lineno', 0), getattr(arg, 'col_offset', 0))