import gast as ast
from pythran.tables import MODULES
from pythran.conversion import mangle, demangle
from functools import reduce
from contextlib import contextmanager
def attr_to_path(node):
    """ Compute path and final object for an attribute node """

    def get_intrinsic_path(modules, attr):
        """ Get function path and intrinsic from an ast.Attribute.  """
        if isinstance(attr, ast.Name):
            return (modules[demangle(attr.id)], (demangle(attr.id),))
        elif isinstance(attr, ast.Attribute):
            module, path = get_intrinsic_path(modules, attr.value)
            return (module[attr.attr], path + (attr.attr,))
    obj, path = get_intrinsic_path(MODULES, node)
    if hasattr(obj, 'isliteral') and (not obj.isliteral()):
        path = path[:-1] + ('functor', path[-1])
    return (obj, ('pythonic',) + path)