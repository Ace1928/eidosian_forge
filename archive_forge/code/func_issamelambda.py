from pythran.analyses import GlobalDeclarations, ImportedIds
from pythran.analyses import Check
from pythran.analyses import ExtendedDefUseChains
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import mangle
import pythran.metadata as metadata
from copy import copy, deepcopy
import gast as ast
def issamelambda(pattern, f1):
    f0, duc = pattern
    if len(f0.args.args) != len(f1.args.args):
        return False
    for arg0, arg1 in zip(f0.args.args, f1.args.args):
        arg0.id = arg1.id
        for u in duc.chains[arg0].users():
            u.node.id = arg1.id
    return Check(f0, {}).visit(f1)