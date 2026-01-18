from pythran.analyses import Aliases
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.intrinsic import FunctionIntr
from pythran.utils import path_to_attr, path_to_node
from pythran.syntax import PythranSyntaxError
from copy import deepcopy
import gast as ast
def make_array_index(self, base, size, index):
    if isinstance(base, ast.Constant):
        return ast.Constant(base.value, None)
    if size == 1:
        return deepcopy(base.elts[0])
    return base.elts[index]