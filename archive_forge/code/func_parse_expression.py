import ast
import inspect
import io
import linecache
import re
import sys
import textwrap
import tokenize
import astunparse
import gast
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.util import tf_inspect
def parse_expression(src):
    """Returns the AST of given identifier.

  Args:
    src: A piece of code that represents a single Python expression
  Returns:
    A gast.AST object.
  Raises:
    ValueError: if src does not consist of a single Expression.
  """
    src = STANDARD_PREAMBLE + src.strip()
    node = parse(src, preamble_len=STANDARD_PREAMBLE_LEN, single_node=True)
    if __debug__:
        if not isinstance(node, gast.Expr):
            raise ValueError('expected exactly one node of type Expr, got {}'.format(node))
    return node.value