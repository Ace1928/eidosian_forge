from pythran.analyses import Globals, Ancestors
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.tables import attributes, functions, methods, MODULES
from pythran.tables import duplicated_methods
from pythran.conversion import mangle, demangle
from pythran.utils import isstr
import gast as ast
from functools import reduce
def rec(path, cur_module):
    """
                    Recursively rename path content looking in matching module.

                    Prefers __module__ to module if it exists.
                    This recursion is done as modules are visited top->bottom
                    while attributes have to be visited bottom->top.
                    """
    err = 'Function path is chained attributes and name'
    assert isinstance(path, (ast.Name, ast.Attribute)), err
    if isinstance(path, ast.Attribute):
        new_node, cur_module = rec(path.value, cur_module)
        new_id, mname = self.renamer(path.attr, cur_module)
        return (ast.Attribute(new_node, new_id, ast.Load()), cur_module[mname])
    else:
        new_id, mname = self.renamer(path.id, cur_module)
        return (ast.Name(new_id, ast.Load(), None, None), cur_module[mname])