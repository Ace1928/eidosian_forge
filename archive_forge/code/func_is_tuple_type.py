import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def is_tuple_type(t):
    pt = prune(t)
    if isinstance(pt, TypeOperator) and pt.name == 'collection':
        st = prune(pt.types[0])
        if isinstance(st, TypeOperator) and st.name == 'traits':
            tt = prune(st.types[0])
            return isinstance(tt, TypeOperator) and tt.name == 'tuple'
    return False