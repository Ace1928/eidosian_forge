from pythran.analyses.aliases import Aliases
from pythran.analyses.argument_effects import ArgumentEffects
from pythran.analyses.identifiers import Identifiers
from pythran.analyses.pure_expressions import PureExpressions
from pythran.passmanager import FunctionAnalysis
from pythran.syntax import PythranSyntaxError
from pythran.utils import get_variable, isattr
import pythran.metadata as md
import pythran.openmp as openmp
import gast as ast
import sys
def func_args_lazyness(self, func_name, args, node):
    for fun in self.aliases[func_name]:
        if isinstance(fun, ast.Call):
            self.func_args_lazyness(fun.args[0], fun.args[1:] + args, node)
        elif fun in self.argument_effects:
            for i, arg in enumerate(self.argument_effects[fun]):
                if arg and len(args) > i:
                    if isinstance(args[i], ast.Name):
                        self.modify(args[i].id)
        elif isinstance(fun, ast.Name):
            continue
        else:
            for arg in args:
                self.modify(arg)