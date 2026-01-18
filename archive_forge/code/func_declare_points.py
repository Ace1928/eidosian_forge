import collections
import warnings
from sympy.external import import_module
def declare_points(self, ctx, i, j=None):
    if '{' in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()
    name2 = 'point_' + name1
    self.symbol_table2.update({name1: name2})
    self.type2.update({name1: 'point'})
    self.write(name2 + ' = ' + "_me.Point('" + name1 + "')\n")