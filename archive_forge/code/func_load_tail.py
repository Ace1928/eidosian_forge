import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def load_tail(self, q, tail):
    self.msgin(4, 'load_tail', q, tail)
    m = q
    while tail:
        i = tail.find('.')
        if i < 0:
            i = len(tail)
        head, tail = (tail[:i], tail[i + 1:])
        mname = '%s.%s' % (m.__name__, head)
        m = self.import_module(head, mname, m)
        if not m:
            self.msgout(4, 'raise ImportError: No module named', mname)
            raise ImportError('No module named ' + mname)
    self.msgout(4, 'load_tail ->', m)
    return m