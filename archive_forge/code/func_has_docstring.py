from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import re
from pasta.augment import errors
from pasta.base import formatting as fmt
def has_docstring(node):
    return hasattr(node, 'body') and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str)