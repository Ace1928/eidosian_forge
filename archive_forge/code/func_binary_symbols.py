from sympy.core import S
from sympy.core.relational import Eq, Ne
from sympy.logic.boolalg import BooleanFunction
from sympy.utilities.misc import func_name
from .sets import Set
@property
def binary_symbols(self):
    return set().union(*[i.binary_symbols for i in self.args[1].args if i.is_Boolean or i.is_Symbol or isinstance(i, (Eq, Ne))])