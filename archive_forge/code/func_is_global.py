from pythran.passmanager import ModuleAnalysis
from pythran.analyses import StrictAliases, ArgumentEffects
from pythran.syntax import PythranSyntaxError
from pythran.intrinsic import ConstantIntr, FunctionIntr
from pythran import metadata
import gast as ast
def is_global(node):
    return isinstance(node, (FunctionIntr, ast.FunctionDef)) or is_global_constant(node)