import ast
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
def copy_clean(node, preserve_annos=None):
    """Creates a deep copy of an AST.

  The copy will not include fields that are prefixed by '__', with the
  exception of user-specified annotations.

  Args:
    node: ast.AST
    preserve_annos: Optional[Set[Hashable]], annotation keys to include in the
        copy
  Returns:
    ast.AST
  """
    return CleanCopier(preserve_annos).copy(node)