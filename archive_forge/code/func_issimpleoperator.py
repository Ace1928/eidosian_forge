from pythran.analyses import GlobalDeclarations, ImportedIds
from pythran.analyses import Check
from pythran.analyses import ExtendedDefUseChains
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import mangle
import pythran.metadata as metadata
from copy import copy, deepcopy
import gast as ast
def issimpleoperator(node):
    if node.args.defaults:
        return None
    body = node.body
    args = node.args.args
    if isinstance(body, ast.UnaryOp) and len(args) == 1:
        if not isinstance(body.operand, ast.Name):
            return None
        return unaryops[type(body.op)]
    if isinstance(body, ast.BinOp) and len(args) == 2:
        if not all((isinstance(op, ast.Name) for op in (body.left, body.right))):
            return None
        if body.left.id != args[0].id or body.right.id != args[1].id:
            return None
        return binaryops[type(body.op)]