import enum
import gast
def dup(node, copy_map, field_name='___pyct_anno'):
    """Recursively copies annotations in an AST tree.

  Args:
    node: ast.AST
    copy_map: Dict[Hashable, Hashable], maps a source anno key to a destination
        key. All annotations with the source key will be copied to identical
        annotations with the destination key.
    field_name: str
  """
    for n in gast.walk(node):
        for k in copy_map:
            if hasanno(n, k, field_name):
                setanno(n, copy_map[k], getanno(n, k, field_name), field_name)