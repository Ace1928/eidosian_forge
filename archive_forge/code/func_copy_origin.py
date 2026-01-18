import collections
import difflib
import io
import os
import tokenize
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.util import tf_inspect
def copy_origin(from_node, to_node):
    """Copies the origin info from a node to another, recursively."""
    origin = anno.Basic.ORIGIN.of(from_node, default=None)
    if origin is None:
        return
    if not isinstance(to_node, (list, tuple)):
        to_node = (to_node,)
    for node in to_node:
        for n in gast.walk(node):
            anno.setanno(n, anno.Basic.ORIGIN, origin)