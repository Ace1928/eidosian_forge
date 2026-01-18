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
def call_return_alias(self, node):

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

    def full_args(func, call):
        args = call.args
        if isinstance(func, ast.FunctionDef):
            extra = len(func.args.args) - len(args)
            if extra:
                tail = [deepcopy(n) for n in func.args.defaults[extra:]]
                for arg in tail:
                    self.visit(arg)
                args = args + tail
        return args
    func = node.func
    aliases = set()
    if node.keywords:
        pass
    elif isinstance(func, ast.Attribute):
        _, signature = methods.get(func.attr, functions.get(func.attr, [(None, None)])[0])
        if signature:
            args = full_args(signature, node)
            aliases = interprocedural_aliases(signature, args)
    elif isinstance(func, ast.Name):
        func_aliases = self.result[func]
        for func_alias in func_aliases:
            if hasattr(func_alias, 'return_alias'):
                args = full_args(func_alias, node)
                aliases.update(interprocedural_aliases(func_alias, args))
            else:
                pass
    for a in aliases:
        if a not in self.result:
            self.add(a)
    return aliases or self.get_unbound_value_set()