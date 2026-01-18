import abc
import ast
import bisect
import sys
import token
from ast import Module
from typing import Iterable, Iterator, List, Optional, Tuple, Any, cast, TYPE_CHECKING
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from .line_numbers import LineNumbers
from .util import (
def supports_tokenless(node=None):
    """
  Returns True if the Python version and the node (if given) are supported by
  the ``get_text*`` methods of ``ASTText`` without falling back to ``ASTTokens``.
  See ``ASTText`` for why this matters.

  The following cases are not supported:

    - Python 3.7 and earlier
    - PyPy
    - ``ast.arguments`` / ``astroid.Arguments``
    - ``ast.withitem``
    - ``astroid.Comprehension``
    - ``astroid.AssignName`` inside ``astroid.Arguments`` or ``astroid.ExceptHandler``
    - The following nodes in Python 3.8 only:
      - ``ast.arg``
      - ``ast.Starred``
      - ``ast.Slice``
      - ``ast.ExtSlice``
      - ``ast.Index``
      - ``ast.keyword``
  """
    return type(node).__name__ not in _unsupported_tokenless_types and (not (not isinstance(node, ast.AST) and node is not None and (type(node).__name__ == 'AssignName' and type(node.parent).__name__ in ('Arguments', 'ExceptHandler')))) and (sys.version_info[:2] >= (3, 8)) and ('pypy' not in sys.version.lower())