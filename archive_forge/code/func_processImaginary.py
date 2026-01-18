import collections
import warnings
from sympy.external import import_module
def processImaginary(self, ctx):
    name = ctx.ID().getText().lower()
    self.symbol_table[name] = name
    self.type[name] = 'imaginary'
    self.var_list.append(name)