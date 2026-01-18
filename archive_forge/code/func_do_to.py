from importlib import import_module
import os
import sys
def do_to(self):
    txt = ["%s'to': {" % self.indent]
    i2 = self.indent + self.indent
    _to = self.mod.MAP['to']
    _keys = _to.keys()
    _keys.sort()
    invmap = {v: k for k, v in self.variable.items()}
    for key in _keys:
        val = _to[key]
        for _urn, _name in invmap.items():
            if val.startswith(_urn):
                txt.append(f"{i2}'{key}': {_name}+'{val[len(_urn):]}',")
    txt.append('%s}' % self.indent)
    return txt