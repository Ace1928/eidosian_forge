import collections
import warnings
from sympy.external import import_module
def processVariables(self, ctx):
    name = ctx.ID().getText().lower()
    if '=' in ctx.getText():
        text = name + "'" * (ctx.getChildCount() - 3)
        self.write(text + ' = ' + self.getValue(ctx.expr()) + '\n')
        return
    if ctx.getChildCount() == 1:
        self.symbol_table[name] = name
        if self.getValue(ctx.parentCtx.getChild(0)) in ('variable', 'specified', 'motionvariable', "motionvariable'"):
            self.type.update({name: self.getValue(ctx.parentCtx.getChild(0))})
        self.var_list.append(name)
        self.sign[name] = 0
    elif "'" in ctx.getText() and '{' not in ctx.getText():
        if ctx.getText().count("'") > self.maxDegree:
            self.maxDegree = ctx.getText().count("'")
        for i in range(ctx.getChildCount()):
            self.sign[name + strfunc(i)] = i
            self.symbol_table[name + "'" * i] = name + strfunc(i)
            if self.getValue(ctx.parentCtx.getChild(0)) in ('variable', 'specified', 'motionvariable', "motionvariable'"):
                self.type.update({name + "'" * i: self.getValue(ctx.parentCtx.getChild(0))})
            self.var_list.append(name + strfunc(i))
    elif '{' in ctx.getText():
        if "'" in ctx.getText():
            dash_count = ctx.getText().count("'")
            if dash_count > self.maxDegree:
                self.maxDegree = dash_count
        if ':' in ctx.getText():
            if ',' in ctx.getText():
                num1 = int(ctx.INT(0).getText())
                num2 = int(ctx.INT(1).getText()) + 1
                num3 = int(ctx.INT(2).getText())
                num4 = int(ctx.INT(3).getText()) + 1
            else:
                num1 = int(ctx.INT(0).getText())
                num2 = int(ctx.INT(1).getText()) + 1
        elif ',' in ctx.getText():
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1
            num3 = 1
            num4 = int(ctx.INT(1).getText()) + 1
        else:
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1
        for i in range(num1, num2):
            try:
                for j in range(num3, num4):
                    try:
                        for z in range(dash_count + 1):
                            self.symbol_table.update({name + str(i) + str(j) + "'" * z: name + str(i) + str(j) + strfunc(z)})
                            if self.getValue(ctx.parentCtx.getChild(0)) in ('variable', 'specified', 'motionvariable', "motionvariable'"):
                                self.type.update({name + str(i) + str(j) + "'" * z: self.getValue(ctx.parentCtx.getChild(0))})
                            self.var_list.append(name + str(i) + str(j) + strfunc(z))
                            self.sign.update({name + str(i) + str(j) + strfunc(z): z})
                            if dash_count > self.maxDegree:
                                self.maxDegree = dash_count
                    except Exception:
                        self.symbol_table.update({name + str(i) + str(j): name + str(i) + str(j)})
                        if self.getValue(ctx.parentCtx.getChild(0)) in ('variable', 'specified', 'motionvariable', "motionvariable'"):
                            self.type.update({name + str(i) + str(j): self.getValue(ctx.parentCtx.getChild(0))})
                        self.var_list.append(name + str(i) + str(j))
                        self.sign.update({name + str(i) + str(j): 0})
            except Exception:
                try:
                    for z in range(dash_count + 1):
                        self.symbol_table.update({name + str(i) + "'" * z: name + str(i) + strfunc(z)})
                        if self.getValue(ctx.parentCtx.getChild(0)) in ('variable', 'specified', 'motionvariable', "motionvariable'"):
                            self.type.update({name + str(i) + "'" * z: self.getValue(ctx.parentCtx.getChild(0))})
                        self.var_list.append(name + str(i) + strfunc(z))
                        self.sign.update({name + str(i) + strfunc(z): z})
                        if dash_count > self.maxDegree:
                            self.maxDegree = dash_count
                except Exception:
                    self.symbol_table.update({name + str(i): name + str(i)})
                    if self.getValue(ctx.parentCtx.getChild(0)) in ('variable', 'specified', 'motionvariable', "motionvariable'"):
                        self.type.update({name + str(i): self.getValue(ctx.parentCtx.getChild(0))})
                    self.var_list.append(name + str(i))
                    self.sign.update({name + str(i): 0})