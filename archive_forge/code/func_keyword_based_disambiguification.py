from pythran.analyses import Globals, Ancestors
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.tables import attributes, functions, methods, MODULES
from pythran.tables import duplicated_methods
from pythran.conversion import mangle, demangle
from pythran.utils import isstr
import gast as ast
from functools import reduce
def keyword_based_disambiguification(self, node):
    assert isinstance(node.func, ast.Attribute)
    if getattr(node.func.value, 'id', None) != mangle('__dispatch__'):
        return
    if not node.keywords:
        return
    if node.func.attr not in duplicated_methods:
        return
    node_keywords = {kw.arg for kw in node.keywords}
    for disamb_path, disamb_node in duplicated_methods[node.func.attr]:
        disamb_args = {arg.id for arg in disamb_node.args.args}
        if all((kw in disamb_args for kw in node_keywords)):
            node.func = self.attr_to_func(node.func, disamb_path)
            return