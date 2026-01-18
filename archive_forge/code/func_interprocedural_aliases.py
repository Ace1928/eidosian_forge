from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.intrinsic import Intrinsic, Class, UnboundValue
from pythran.passmanager import ModuleAnalysis
from pythran.tables import functions, methods, MODULES
from pythran.unparse import Unparser
from pythran.conversion import demangle
import pythran.metadata as md
from pythran.utils import isnum
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
from itertools import product
import io
def interprocedural_aliases(func, args):
    arg_aliases = [self.result[arg] or {arg} for arg in args]
    return_aliases = set()
    for args_combination in product(*arg_aliases):
        for ra in func.return_alias(args_combination):
            if isinstance(ra, ast.Subscript):
                if isinstance(ra.value, ContainerOf):
                    return_aliases.update(ra.value.containees)
                    continue
            return_aliases.add(ra)
    return return_aliases