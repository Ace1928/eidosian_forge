from pythran.passmanager import ModuleAnalysis
from pythran.analyses import StrictAliases, ArgumentEffects
from pythran.syntax import PythranSyntaxError
from pythran.intrinsic import ConstantIntr, FunctionIntr
from pythran import metadata
import gast as ast
def is_immutable_constant(self, node):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Tuple):
        return all((self.is_immutable_constant(elt) for elt in node.elts))
    if isinstance(node, ast.UnaryOp):
        return self.is_immutable_constant(node.operand)
    if isinstance(node, ast.Call):
        target = getattr(node, 'func', node)
        try:
            aliases = self.strict_aliases[target]
        except KeyError:
            return False
        if not aliases:
            return False
        if all((is_global_constant(alias) for alias in aliases)):
            return True
    if isinstance(node, ast.Attribute):
        target = getattr(node, 'func', node)
        try:
            aliases = self.strict_aliases[target]
        except KeyError:
            return False
        if not aliases:
            return False
        if all((is_global(alias) for alias in aliases)):
            return True
    if isinstance(node, ast.Name):
        try:
            aliases = self.strict_aliases[node]
        except KeyError:
            return False
        if all((isinstance(alias, ast.FunctionDef) for alias in aliases)):
            return True
    return False