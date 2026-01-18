from pythran.analyses.aliases import Aliases
from pythran.analyses.argument_effects import ArgumentEffects
from pythran.analyses.identifiers import Identifiers
from pythran.analyses.pure_expressions import PureExpressions
from pythran.passmanager import FunctionAnalysis
from pythran.syntax import PythranSyntaxError
from pythran.utils import get_variable, isattr
import pythran.metadata as md
import pythran.openmp as openmp
import gast as ast
import sys
def assign_to(self, node, from_):
    if isinstance(node, ast.Name):
        self.name_to_nodes.setdefault(node.id, set()).add(node)
    if node.id in self.dead:
        self.dead.remove(node.id)
    self.result[node.id] = max(self.result.get(node.id, 0), self.name_count.get(node.id, 0))
    self.in_omp.discard(node.id)
    pre_loop = self.pre_loop_count.setdefault(node.id, (0, True))
    if not pre_loop[1]:
        self.pre_loop_count[node.id] = (pre_loop[0], True)
    self.modify(node.id)
    self.name_count[node.id] = 0
    self.use[node.id] = set(from_)