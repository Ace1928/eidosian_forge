import sys
import re
import copy
import time
import os.path
def macro_prescan(self, macro):
    macro.patch = []
    macro.str_patch = []
    macro.var_comma_patch = []
    i = 0
    while i < len(macro.value):
        if macro.value[i].type == self.t_ID and macro.value[i].value in macro.arglist:
            argnum = macro.arglist.index(macro.value[i].value)
            if i > 0 and macro.value[i - 1].value == '#':
                macro.value[i] = copy.copy(macro.value[i])
                macro.value[i].type = self.t_STRING
                del macro.value[i - 1]
                macro.str_patch.append((argnum, i - 1))
                continue
            elif i > 0 and macro.value[i - 1].value == '##':
                macro.patch.append(('c', argnum, i - 1))
                del macro.value[i - 1]
                continue
            elif i + 1 < len(macro.value) and macro.value[i + 1].value == '##':
                macro.patch.append(('c', argnum, i))
                i += 1
                continue
            else:
                macro.patch.append(('e', argnum, i))
        elif macro.value[i].value == '##':
            if macro.variadic and i > 0 and (macro.value[i - 1].value == ',') and (i + 1 < len(macro.value)) and (macro.value[i + 1].type == self.t_ID) and (macro.value[i + 1].value == macro.vararg):
                macro.var_comma_patch.append(i - 1)
        i += 1
    macro.patch.sort(key=lambda x: x[2], reverse=True)