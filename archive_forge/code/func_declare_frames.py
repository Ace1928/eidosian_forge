import collections
import warnings
from sympy.external import import_module
def declare_frames(self, ctx, i, j=None):
    if '{' in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()
    name2 = 'frame_' + name1
    if self.getValue(ctx.parentCtx.varType()) == 'newtonian':
        self.newtonian = name2
    self.symbol_table2.update({name1: name2})
    self.symbol_table.update({name1 + '1>': name2 + '.x'})
    self.symbol_table.update({name1 + '2>': name2 + '.y'})
    self.symbol_table.update({name1 + '3>': name2 + '.z'})
    self.type2.update({name1: 'frame'})
    self.write(name2 + ' = ' + "_me.ReferenceFrame('" + name1 + "')\n")