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
def check_slice_includes_step(self, node):
    """Helper function for Slice node to determine whether to visit its step."""
    return self.tokens.peek_non_whitespace().src not in '],'