import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def full_name_node(name, ctx=ast.Load()):
    """Make an Attribute or Name node for name.

  Translate a qualified name into nested Attribute nodes (and a Name node).

  Args:
    name: The name to translate to a node.
    ctx: What context this name is used in. Defaults to Load()

  Returns:
    A Name or Attribute node.
  """
    names = name.split('.')
    names.reverse()
    node = ast.Name(id=names.pop(), ctx=ast.Load())
    while names:
        node = ast.Attribute(value=node, attr=names.pop(), ctx=ast.Load())
    node.ctx = ctx
    return node