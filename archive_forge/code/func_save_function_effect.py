from pythran.analyses.aliases import Aliases
from pythran.analyses.intrinsics import Intrinsics
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
from pythran.graph import DiGraph
from pythran import intrinsic
import gast as ast
from functools import reduce
def save_function_effect(module):
    """ Recursively save function effect for pythonic functions. """
    for intr in module.values():
        if isinstance(intr, dict):
            save_function_effect(intr)
        else:
            fe = FunctionEffects(intr)
            IntrinsicArgumentEffects[intr] = fe
            if isinstance(intr, intrinsic.Class):
                save_function_effect(intr.fields)