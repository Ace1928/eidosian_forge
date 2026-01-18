from pythran.analyses.aliases import Aliases
from pythran.analyses.intrinsics import Intrinsics
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
from pythran.graph import DiGraph
import pythran.intrinsic as intrinsic
import gast as ast
from functools import reduce
def save_global_effects(module):
    """ Recursively save globals effect for all functions. """
    for intr in module.values():
        if isinstance(intr, dict):
            save_global_effects(intr)
        else:
            fe = FunctionEffect(intr)
            IntrinsicGlobalEffects[intr] = fe
            if isinstance(intr, intrinsic.Class):
                save_global_effects(intr.fields)