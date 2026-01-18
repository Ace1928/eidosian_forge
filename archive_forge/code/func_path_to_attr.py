import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
def path_to_attr(path):
    """
    Transform path to ast.Attribute.

    >>> import gast as ast
    >>> path = ('builtins', 'my', 'constant')
    >>> value = path_to_attr(path)
    >>> ref = ast.Attribute(
    ...     value=ast.Attribute(value=ast.Name(id="builtins",
    ...                                        ctx=ast.Load(),
    ...                                        annotation=None,
    ...                                        type_comment=None),
    ...                         attr="my", ctx=ast.Load()),
    ...     attr="constant", ctx=ast.Load())
    >>> ast.dump(ref) == ast.dump(value)
    True
    """
    return reduce(lambda hpath, last: ast.Attribute(hpath, last, ast.Load()), path[1:], ast.Name(mangle(path[0]), ast.Load(), None, None))