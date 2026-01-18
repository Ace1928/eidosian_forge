import ast
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
def apply_to_single_assignments(targets, values, apply_fn):
    """Applies a function to each individual assignment.

  This function can process a possibly-unpacked (e.g. a, b = c, d) assignment.
  It tries to break down the unpacking if possible. In effect, it has the same
  effect as passing the assigned values in SSA form to apply_fn.

  Examples:

  The following will result in apply_fn(a, c), apply_fn(b, d):

      a, b = c, d

  The following will result in apply_fn(a, c[0]), apply_fn(b, c[1]):

      a, b = c

  The following will result in apply_fn(a, (b, c)):

      a = b, c

  It uses the visitor pattern to allow subclasses to process single
  assignments individually.

  Args:
    targets: Union[List[ast.AST, ...], Tuple[ast.AST, ...], ast.AST, should be
        used with the targets field of an ast.Assign node
    values: ast.AST
    apply_fn: Callable[[ast.AST, ast.AST], None], called with the
        respective nodes of each single assignment
  """
    if not isinstance(targets, (list, tuple)):
        targets = (targets,)
    for target in targets:
        if isinstance(target, (gast.Tuple, gast.List)):
            for i in range(len(target.elts)):
                target_el = target.elts[i]
                if isinstance(values, (gast.Tuple, gast.List)):
                    value_el = values.elts[i]
                else:
                    idx = parser.parse_expression(str(i))
                    value_el = gast.Subscript(values, idx, ctx=gast.Load())
                apply_to_single_assignments(target_el, value_el, apply_fn)
        else:
            apply_fn(target, values)