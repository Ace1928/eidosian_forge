from importlib import import_module
import os
import sys
def do_fro(self):
    txt = ["%s'fro': {" % self.indent]
    i2 = self.indent + self.indent
    _fro = self.mod.MAP['fro']
    for var in self.vars:
        _v = self.variable[var]
        li = [k[len(_v):] for k in _fro.keys() if k.startswith(_v)]
        li.sort(intcmp)
        for item in li:
            txt.append(f"{i2}{var}+'{item}': '{_fro[_v + item]}',")
    txt.append('%s},' % self.indent)
    return txt