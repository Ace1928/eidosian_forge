import ast
import textwrap
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
def replace_as_expression(template, **replacements):
    """Variant of replace that generates expressions, instead of code blocks."""
    replacement = replace(template, **replacements)
    if len(replacement) != 1:
        raise ValueError('single expression expected; for more general templates use replace')
    node, = replacement
    if isinstance(node, gast.Expr):
        return node.value
    elif isinstance(node, gast.Name):
        return node
    raise ValueError('the template is expected to generate an expression or a name node; instead found %s' % node)