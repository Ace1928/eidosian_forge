from pythran.passmanager import ModuleAnalysis
from pythran.analyses import StrictAliases, ArgumentEffects
from pythran.syntax import PythranSyntaxError
from pythran.intrinsic import ConstantIntr, FunctionIntr
from pythran import metadata
import gast as ast
def check_global_with_side_effect(self, node, arg):
    if not isinstance(arg, ast.Call):
        return
    try:
        aliases = self.strict_aliases[arg.func]
    except KeyError:
        return
    for alias in aliases:
        if is_global_constant(alias):
            raise PythranSyntaxError("Cannot modify '{}': global variables are constant in pythran.".format(alias.name), arg.func)