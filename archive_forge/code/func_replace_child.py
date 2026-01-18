from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import re
from pasta.augment import errors
from pasta.base import formatting as fmt
def replace_child(parent, node, replace_with):
    """Replace a node's child with another node while preserving formatting.

  Arguments:
    parent: (ast.AST) Parent node to replace a child of.
    node: (ast.AST) Child node to replace.
    replace_with: (ast.AST) New child node.
  """
    if hasattr(node, fmt.PASTA_DICT):
        fmt.set(replace_with, 'prefix', fmt.get(node, 'prefix'))
        fmt.set(replace_with, 'suffix', fmt.get(node, 'suffix'))
    for field in parent._fields:
        field_val = getattr(parent, field, None)
        if field_val == node:
            setattr(parent, field, replace_with)
            return
        elif isinstance(field_val, list):
            try:
                field_val[field_val.index(node)] = replace_with
                return
            except ValueError:
                pass
    raise errors.InvalidAstError('Node %r is not a child of %r' % (node, parent))