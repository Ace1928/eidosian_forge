from pythran.analyses.aliases import Aliases
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
import pythran.intrinsic as intrinsic
import gast as ast
from functools import reduce
def local_effect(self, node, effect):
    index = self.argument_index(node)
    return lambda ctx: effect if index == ctx.index else 0