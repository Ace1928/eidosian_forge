import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def is_option_type(t):
    pt = prune(t)
    return isinstance(pt, TypeOperator) and pt.name == 'option'