from pythran.analyses import Ancestors
from pythran.passmanager import Transformation
import gast as ast
def check_lasts(self, node):
    if isinstance(node, (ast.Return, ast.Break, ast.Return)):
        return True
    if isinstance(node, ast.If):
        if not self.check_lasts(node.body[-1]):
            return False
        return node.orelse and self.check_lasts(node.orelse[-1])