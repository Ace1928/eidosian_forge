from pythran.analyses import Aliases
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.intrinsic import FunctionIntr
from pythran.utils import path_to_attr, path_to_node
from pythran.syntax import PythranSyntaxError
from copy import deepcopy
import gast as ast
def inlineFixedSizeArrayCompare(self, node):
    if len(node.comparators) != 1:
        return node
    node_right = node.comparators[0]
    alike = (ast.Constant, ast.List, ast.Tuple)
    if isinstance(node.left, alike) and isinstance(node_right, alike):
        return node
    lbase, lsize = self.fixedSizeArray(node.left)
    rbase, rsize = self.fixedSizeArray(node_right)
    if not lbase or not rbase:
        return node
    if rsize != 1 and lsize != 1 and (rsize != lsize):
        raise PythranSyntaxError('Invalid numpy broadcasting', node)
    self.update = True
    operands = [ast.Compare(self.make_array_index(lbase, lsize, i), [type(node.ops[0])()], [self.make_array_index(rbase, rsize, i)]) for i in range(max(lsize, rsize))]
    res = ast.Call(path_to_attr(('numpy', 'array')), [ast.Tuple(operands, ast.Load())], [])
    self.aliases[res.func] = {path_to_node(('numpy', 'array'))}
    return res