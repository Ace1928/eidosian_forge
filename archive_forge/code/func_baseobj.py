from pythran.analyses import Globals, Ancestors
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.tables import attributes, functions, methods, MODULES
from pythran.tables import duplicated_methods
from pythran.conversion import mangle, demangle
from pythran.utils import isstr
import gast as ast
from functools import reduce
def baseobj(self, obj):
    while isinstance(obj, ast.Attribute):
        obj = obj.value
    if isinstance(obj, ast.Name) and obj.id in self.imports:
        return None
    else:
        return obj