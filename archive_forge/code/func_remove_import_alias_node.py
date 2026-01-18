from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import copy
import logging
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import scope
def remove_import_alias_node(sc, node):
    """Remove an alias and if applicable remove their entire import.

  Arguments:
    sc: (scope.Scope) Scope computed on whole tree of the code being modified.
    node: (ast.Import|ast.ImportFrom|ast.alias) The node to remove.
  """
    import_node = sc.parent(node)
    if len(import_node.names) == 1:
        import_parent = sc.parent(import_node)
        ast_utils.remove_child(import_parent, import_node)
    else:
        ast_utils.remove_child(import_node, node)