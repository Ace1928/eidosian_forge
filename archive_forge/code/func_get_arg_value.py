import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def get_arg_value(node, arg_name, arg_pos=None):
    """Get the value of an argument from a ast.Call node.

  This function goes through the positional and keyword arguments to check
  whether a given argument was used, and if so, returns its value (the node
  representing its value).

  This cannot introspect *args or **args, but it safely handles *args in
  Python3.5+.

  Args:
    node: The ast.Call node to extract arg values from.
    arg_name: The name of the argument to extract.
    arg_pos: The position of the argument (in case it's passed as a positional
      argument).

  Returns:
    A tuple (arg_present, arg_value) containing a boolean indicating whether
    the argument is present, and its value in case it is.
  """
    if arg_name is not None:
        for kw in node.keywords:
            if kw.arg == arg_name:
                return (True, kw.value)
    if arg_pos is not None:
        idx = 0
        for arg in node.args:
            if sys.version_info[:2] >= (3, 5) and isinstance(arg, ast.Starred):
                continue
            if idx == arg_pos:
                return (True, arg)
            idx += 1
    return (False, None)