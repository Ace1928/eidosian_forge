from pythran.analyses.aliases import Aliases
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
import pythran.intrinsic as intrinsic
import gast as ast
from functools import reduce
def argument_index(self, node):
    while isinstance(node, ast.Subscript):
        node = node.value
    if node in self.aliases:
        for n_alias in self.aliases[node]:
            try:
                return self.current_function.func.args.args.index(n_alias)
            except ValueError:
                pass
    return -1