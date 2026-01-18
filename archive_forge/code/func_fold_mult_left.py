from pythran.analyses import ConstantExpressions, ASTMatcher
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import to_ast, ConversionError, ToNotEval, mangle
from pythran.analyses.ast_matcher import DamnTooLongPattern
from pythran.syntax import PythranSyntaxError
from pythran.utils import isintegral, isnum
from pythran.config import cfg
import builtins
import gast as ast
from copy import deepcopy
import logging
import sys
def fold_mult_left(self, node):
    if not isinstance(node.left, (ast.List, ast.Tuple)):
        return False
    if not isnum(node.right):
        return False
    if not isintegral(node.right):
        raise PythranSyntaxError('Multiplying a sequence by a float', node)
    return isinstance(node.op, ast.Mult)