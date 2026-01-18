from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import re
from pasta.augment import errors
from pasta.base import formatting as fmt
def get_last_child(node):
    """Get the last child node of a block statement.

  The input must be a block statement (e.g. ast.For, ast.With, etc).

  Examples:
    1. with first():
         second()
         last()

    2. try:
         first()
       except:
         second()
       finally:
         last()

  In both cases, the last child is the node for `last`.
  """
    if isinstance(node, ast.Module):
        try:
            return node.body[-1]
        except IndexError:
            return None
    if isinstance(node, ast.If):
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If) and fmt.get(node.orelse[0], 'is_elif'):
            return get_last_child(node.orelse[0])
        if node.orelse:
            return node.orelse[-1]
    elif isinstance(node, ast.With):
        if len(node.body) == 1 and isinstance(node.body[0], ast.With) and fmt.get(node.body[0], 'is_continued'):
            return get_last_child(node.body[0])
    elif hasattr(ast, 'Try') and isinstance(node, ast.Try):
        if node.finalbody:
            return node.finalbody[-1]
        if node.orelse:
            return node.orelse[-1]
    elif hasattr(ast, 'TryFinally') and isinstance(node, ast.TryFinally):
        if node.finalbody:
            return node.finalbody[-1]
    elif hasattr(ast, 'TryExcept') and isinstance(node, ast.TryExcept):
        if node.orelse:
            return node.orelse[-1]
        if node.handlers:
            return get_last_child(node.handlers[-1])
    return node.body[-1]