import operator
import sys
from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import NondifferentiableError
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import value, native_types
def getSympySymbol(self, pyomo_object):
    if pyomo_object in self.pyomo2sympy:
        return self.pyomo2sympy[pyomo_object]
    sympy_obj = sympy.Symbol('x%d' % self.i, real=True)
    self.i += 1
    self.pyomo2sympy[pyomo_object] = sympy_obj
    self.sympy2pyomo[sympy_obj] = pyomo_object
    return sympy_obj