import collections
import warnings
from sympy.external import import_module
def processConstants(self, ctx):
    name = ctx.ID().getText().lower()
    if '=' in ctx.getText():
        self.symbol_table.update({name: name})
        self.write(self.symbol_table[name] + ' = ' + '_sm.S(' + self.getValue(ctx.getChild(2)) + ')\n')
        self.type.update({name: 'constants'})
        return
    elif '{' not in ctx.getText():
        self.symbol_table[name] = name
        self.type[name] = 'constants'
    if ctx.getChildCount() == 2:
        if ctx.getChild(1).getText() == '+':
            self.sign[name] = '+'
        elif ctx.getChild(1).getText() == '-':
            self.sign[name] = '-'
    elif '{' not in ctx.getText():
        self.sign[name] = 'o'
    if '{' in ctx.getText():
        if ':' in ctx.getText():
            num1 = int(ctx.INT(0).getText())
            num2 = int(ctx.INT(1).getText()) + 1
        else:
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1
        if ':' in ctx.getText():
            if ',' in ctx.getText():
                num3 = int(ctx.INT(2).getText())
                num4 = int(ctx.INT(3).getText()) + 1
                for i in range(num1, num2):
                    for j in range(num3, num4):
                        self.symbol_table[name + str(i) + str(j)] = name + str(i) + str(j)
                        self.type[name + str(i) + str(j)] = 'constants'
                        self.var_list.append(name + str(i) + str(j))
                        self.sign[name + str(i) + str(j)] = 'o'
            else:
                for i in range(num1, num2):
                    self.symbol_table[name + str(i)] = name + str(i)
                    self.type[name + str(i)] = 'constants'
                    self.var_list.append(name + str(i))
                    self.sign[name + str(i)] = 'o'
        elif ',' in ctx.getText():
            for i in range(1, int(ctx.INT(0).getText()) + 1):
                for j in range(1, int(ctx.INT(1).getText()) + 1):
                    self.symbol_table[name] = name + str(i) + str(j)
                    self.type[name + str(i) + str(j)] = 'constants'
                    self.var_list.append(name + str(i) + str(j))
                    self.sign[name + str(i) + str(j)] = 'o'
        else:
            for i in range(num1, num2):
                self.symbol_table[name + str(i)] = name + str(i)
                self.type[name + str(i)] = 'constants'
                self.var_list.append(name + str(i))
                self.sign[name + str(i)] = 'o'
    if '{' not in ctx.getText():
        self.var_list.append(name)