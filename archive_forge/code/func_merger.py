from pythran.analyses.aliases import Aliases
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
import pythran.intrinsic as intrinsic
import gast as ast
from functools import reduce
def merger(ctx):
    base = l0(ctx)
    if ctx.index in index_corres and ctx.global_dependencies:
        rec = self.recursive_weight(func, index_corres[ctx.index], ctx.path)
    else:
        rec = 0
    return base + rec