import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def scan_code(self, co, m):
    code = co.co_code
    scanner = self.scan_opcodes
    for what, args in scanner(co):
        if what == 'store':
            name, = args
            m.globalnames[name] = 1
        elif what == 'absolute_import':
            fromlist, name = args
            have_star = 0
            if fromlist is not None:
                if '*' in fromlist:
                    have_star = 1
                fromlist = [f for f in fromlist if f != '*']
            self._safe_import_hook(name, m, fromlist, level=0)
            if have_star:
                mm = None
                if m.__path__:
                    mm = self.modules.get(m.__name__ + '.' + name)
                if mm is None:
                    mm = self.modules.get(name)
                if mm is not None:
                    m.globalnames.update(mm.globalnames)
                    m.starimports.update(mm.starimports)
                    if mm.__code__ is None:
                        m.starimports[name] = 1
                else:
                    m.starimports[name] = 1
        elif what == 'relative_import':
            level, fromlist, name = args
            if name:
                self._safe_import_hook(name, m, fromlist, level=level)
            else:
                parent = self.determine_parent(m, level=level)
                self._safe_import_hook(parent.__name__, None, fromlist, level=0)
        else:
            raise RuntimeError(what)
    for c in co.co_consts:
        if isinstance(c, type(co)):
            self.scan_code(c, m)